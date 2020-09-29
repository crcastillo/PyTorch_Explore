"""
    PyTorch Explore - Classification Example

    Objective: Construct various Classification Models within PyTorch framework and see if it can be stood up on GPU.

    Initial Build: 9/15/2020

    Notes:
    -
"""

# <editor-fold desc="Import relevant modules, load dataset, split Test/Train/Valid">

import torch
import pytorch_lightning as pl

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker

import optuna
import multiprocessing
import importlib
import datetime

import plotly.express as px
# from datetime import datetime
# import pickle


# Set parameters
Random_Seed = 123
Test_Proportion = 0.2
Cores = np.int(multiprocessing.cpu_count() / 2)
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64

# Import PreProcessing module
MyModule = importlib.import_module('PreProcessing')
# MyModule = importlib.reload(importlib.import_module('PreProcessing'))
PreProcessing = MyModule.Pipeline_PreProcessing

# Import bank-marketing dataset, target is whether client subscribed to a term deposit
Data_X, Data_Y = fetch_openml(
    data_id=1461
    , return_X_y=True
    , as_frame=True
)

# Reset the target levels from ['1', '2']
Data_Y.dtype._categories = ['0', '1']

# Name the columns
Data_X.columns = [
    'age'
    , 'job'
    , 'marital'
    , 'education'
    , 'default'
    , 'balance'
    , 'housing'
    , 'loan'
    , 'contact'
    , 'day'
    , 'month'
    , 'duration'
    , 'campaign'
    , 'pdays'
    , 'previous'
    , 'poutcome'
]

# Split Data into Test/Train
X_Train, X_Test, Y_Train, Y_Test = train_test_split(
    Data_X
    , Data_Y
    , random_state = Random_Seed
    , test_size = Test_Proportion
)

# Split Train into Train_Split/Valid_Split
X_Train_Split, X_Valid_Split, Y_Train_Split, Y_Valid_Split = train_test_split(
    X_Train
    , Y_Train
    , random_state = Random_Seed
    , test_size = Test_Proportion
)

# Fit and transform feature matrices
x_train = PreProcessing.fit_transform(X = X_Train_Split).astype('float32')
x_valid = PreProcessing.transform(X = X_Valid_Split).astype('float32')
x_test = PreProcessing.transform(X = X_Test).astype('float32')

# Define training data class
class trainData(torch.utils.data.Dataset):
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]
    def __len__(self):
        return len(self.x_train)

# Define Data_Transform class to use in DataLoader
class Data_Transform(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.X = self.X.astype('float32')
        self.y = y
        self.y = self.y.astype('float32')
        self.y = self.y.values.reshape((len(self.y), 1))
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    def __len__(self):
        return len(self.X)

# Convert into Datasets
train_data = Data_Transform(X=x_train, y=Y_Train_Split)
valid_data = Data_Transform(X=x_valid, y=Y_Valid_Split)
test_data = Data_Transform(X=x_test, y=Y_Test)

# Create the DataLoaders
train_dl = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_dl = torch.utils.data.DataLoader(dataset=valid_data, batch_size=32, shuffle=False)
test_dl = torch.utils.data.DataLoader(dataset=test_data, batch_size=32, shuffle=False)

# </editor-fold>

# <editor-fold desc="Create NeuralNet class and define model architecture">

# Inherit the basic NeuralNetwork module
class NeuralNet(torch.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.Linear1 = torch.nn.modules.linear.Linear(
            in_features=x_train.shape[1]
            , out_features=42
        )
        self.Linear2 = torch.nn.modules.linear.Linear(
            in_features=42
            , out_features=21
        )
        self.Linear3 = torch.nn.modules.linear.Linear(
            in_features=21
            , out_features=1
        )
    # Define the forward pass output
    def forward(self, x):
        x = torch.nn.functional.relu(self.Linear1(x))
        x = torch.nn.functional.relu(self.Linear2(x))
        return torch.sigmoid(self.Linear3(x))

# </editor-fold>

# <editor-fold desc="Define model parameters and train">

# Instantiate the model
NN_Model = NeuralNet()

# Define the model parameters
epochs = 1000
learning_rate = 1e-3
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(
    params=NN_Model.parameters()
    ,lr=learning_rate
)

# If GPU available then instantiate model on GPU
NN_Model.to(Device)

# Store the initial start time
StartTime = datetime.datetime.now()

# Train the model
Train_BCE_Loss = [] # Instantiate BCE_Loss for model training
Valid_BCE_Loss = [] # Instantiate BCE_Loss for model validation
# Iterate over the epochs
for _e in range(epochs):
    # Instantiate lists to store train/valid loss
    epoch_train_loss = []
    epoch_valid_loss = []
    # Run in batches
    for _inputs, _targets in train_dl:
        # Ensure model is in train
        NN_Model.train()
        # Load inputs to GPU
        _inputs_t = _inputs.to(Device)
        _targets_t = _targets.to(Device)
        # Clear the gradients
        optimizer.zero_grad()
        # Compute model output
        _yhat = NN_Model(_inputs_t)
        # Calculate the loss
        train_loss = criterion(_yhat, _targets_t)
        # Bakward pass
        train_loss.backward()
        # Update model weights
        optimizer.step()
        # Append train_loss
        epoch_train_loss.append(train_loss.item())

    # Validation process
    with torch.set_grad_enabled(False):
        for _inputs, _targets in valid_dl:
            # Ensure model is in evaluate
            NN_Model.eval()
            # Load inputs to GPU
            _inputs_t = _inputs.to(Device)
            _targets_t = _targets.to(Device)
            # Compute model output
            _yhat = NN_Model(_inputs_t)
            # Calculate the loss
            valid_loss = criterion(_yhat, _targets_t)
            # Append train_loss
            epoch_valid_loss.append(valid_loss.item())

    # Append mean loss stats
    Train_BCE_Loss.append(np.mean(epoch_train_loss))
    Valid_BCE_Loss.append(np.mean(epoch_valid_loss))

    # Print epoch statistics
    print(
        f'''epoch {_e}, Train_Loss = {Train_BCE_Loss[-1]}, Valid_Loss {Valid_BCE_Loss[-1]}'''
    )

# Display the runtime
print(datetime.datetime.now() - StartTime)

# Save the model
torch.save(obj=NN_Model.state_dict(), f='./Classify_Example_net.sav')

# Plot the train vs valid loss
px.line(
    data_frame=pd.DataFrame(
        data=list(zip(range(0,epochs), Train_BCE_Loss, Valid_BCE_Loss))
        , columns=['Epoch', 'Train_Loss', 'Valid_Loss']
    )
    , x='Epoch'
    , y=['Train_Loss', 'Valid_Loss']
    , labels = dict(value='Binary Cross Entropy Loss (Sum)', variable='Legend')
    , render_mode='browser'
).show()

# Score Test data
Pred = np.array([])
Actual = np.array([])
with torch.no_grad():
    for _inputs, _targets in test_dl:
        # Ensure model is in evaluate
        NN_Model.eval()
        # Load inputs to GPU
        _inputs_t = _inputs.to(Device)
        # Compute model output
        _yhat = NN_Model(_inputs_t)
        # Append predictions and targets
        Pred = np.append(Pred, _yhat.cpu().data.numpy())
        Actual = np.append(Actual, _targets.cpu().data.numpy())
# Determine AUC and Average Precision Score
print('AUC = {:.4f}'.format(roc_auc_score(y_true=Actual, y_score=Pred)))
print('Average Precision Score = {:.4f}'.format(average_precision_score(y_true=Actual, y_score=Pred)))

# </editor-fold>

# <editor-fold desc="Recreate the model with PyTorch-Lightning Module">

# Create the class
class NN_Lit_Model(pl.LightningModule):
    # Initialize
    def __init__(self):
        super().__init__()
        self.Linear1 = torch.nn.modules.linear.Linear(
            in_features=x_train.shape[1]
            , out_features=42
        )
        self.Linear2 = torch.nn.modules.linear.Linear(
            in_features=42
            , out_features=21
        )
        self.Linear3 = torch.nn.modules.linear.Linear(
            in_features=21
            , out_features=1
        )
    # Define the forward pass output
    def forward(self, x):
        x = torch.nn.functional.relu(self.Linear1(x))
        x = torch.nn.functional.relu(self.Linear2(x))
        x = torch.sigmoid(self.Linear3(x))
        return x
    # Define optimizer
    def configure_optimizers(self):
        return torch.optim.SGD(params=self.parameters(),lr=1e-2)
    # Define the training step
    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = torch.nn.functional.binary_cross_entropy(input=self(x), target=y)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss)
        return result
    # Define the validation step
    def validation_step(self, batch, batch_nb):
        x, y = batch
        loss = torch.nn.functional.binary_cross_entropy(input=self(x), target=y)
        result = pl.EvalResult(
            checkpoint_on=loss
            , early_stop_on=loss
        )
        result.log('val_loss', loss)
        return result

# Instantiate the model
NN_Lit_Model_Test = NN_Lit_Model()

# Instantiate the Trainer
Trainer = pl.Trainer(
    gpus=[Device]
    , max_epochs=1000
    , early_stop_callback=pl.callbacks.EarlyStopping(
        patience=10
        , verbose=True
        # , monitor='val_loss' # Not necessary since specified in validation_step
    )
    , checkpoint_callback=pl.callbacks.ModelCheckpoint(
        filepath='Classifier Checkpoint/'
        # , monitor='val_loss'
        , verbose=True
        , save_top_k=3
    )
    , progress_bar_refresh_rate=1
    , default_root_dir='Classifier Checkpoint/'
)

# Fit the model
Trainer.fit(
    model=NN_Lit_Model_Test
    , train_dataloader=train_dl
    , val_dataloaders=valid_dl
)

# Save trained model object
NN_Lit_Model_trained = NN_Lit_Model_Test

# Save the configuration for inference/prediction
NN_Lit_Model_trained.freeze()

# TODO: Figure out the source of the illegal memory access error

# Make sure we can get a prediction out | success
NN_Lit_Model_trained(torch.randn(42).to(Device))

# Score Test data
Pred = np.array([])
Actual = np.array([])
for _inputs, _targets in test_dl:
    # Load inputs to GPU
    _inputs_t = _inputs.to(Device)
    # Compute model output
    _yhat = NN_Lit_Model_trained(_inputs_t)
    # Append predictions and targets
    Pred = np.append(Pred, _yhat.cpu().data.numpy())
    Actual = np.append(Actual, _targets.cpu().data.numpy())
# Determine AUC and Average Precision Score
print('AUC = {:.4f}'.format(roc_auc_score(y_true=Actual, y_score=Pred)))
print('Average Precision Score = {:.4f}'.format(average_precision_score(y_true=Actual, y_score=Pred)))

# </editor-fold>

# <editor-fold desc="Use Optuna to hypertune the LogisticHazard model and create predictions">



# Simplest build, likely best
# Can't seem to remove duplicated unnecessary weights file...
# Create Callback class to attempt end of epoch trial.prune
class Callback_Prune(tt.callbacks.Callback):
    def __init__(self, trial: optuna.trial.Trial, metric='loss'):
        self.trial = trial
        self.epoch = -1
        self.metric = metric

    def on_epoch_end(self):
        # Advance _epoch by 1
        self.epoch += 1
        # Retrieve the current val score
        self.score = self.model.log.monitors['val_'].scores[self.metric]['score'][-1]
        # Report score to trial
        self.trial.report(value=self.score, step=self.epoch)
        # Check whether should prune
        self.should_prune = self.trial.should_prune()
        # Run pruning boolean
        if self.should_prune:
            # print(self.file_path)
            message = 'Trial was pruned at iteration {}.'.format(self.epoch)
            raise optuna.TrialPruned(message)
        return self.should_prune


# Define function that will optimize hidden layers, in/out features, dropout ratio
def Define_Model(trial: optuna.trial.Trial):

    # Hidden layers between 1 and 5
    n_hidden_layers = trial.suggest_int('n_hidden_layers', 1, 4)

    # Instantiate list for layers
    hidden_layers = []

    # Define starting point for input features
    input_features = X_Train_Split.shape[1]

    # Iterate through hidden layer options
    for _i in range(n_hidden_layers):

        # Store potential out_features
        out_features = trial.suggest_int('n_units_l{}'.format(_i), 4, 112)

        # Append the layer
        hidden_layers.append(
            torch.nn.modules.linear.Linear(
                in_features = input_features
                , out_features = out_features
            )
        )
        # Append the activation function
        hidden_layers.append(torch.nn.modules.activation.ReLU())
        # Append the batch normalization
        hidden_layers.append(torch.nn.modules.batchnorm.BatchNorm1d(num_features = out_features))
        # Suggest p probability for Dropout layer
        p = trial.suggest_uniform("dropout_l{}".format(_i), 0.05, 0.5)
        # Dropout layer, which randomly zeroes some proportion of the elements
        hidden_layers.append(torch.nn.modules.dropout.Dropout(p = p))

        # Set input_features accordingly so any new layers take on the input dimension from the last output dimension
        input_features = out_features

    # Append the final layer
    hidden_layers.append(
        torch.nn.modules.linear.Linear(
            in_features = input_features
            , out_features = Label_Transform.out_features
        )
    )

    # Return back the Sequential Neural Network
    return torch.nn.Sequential(*hidden_layers)


# Define objective function to optimize within Optuna study
def LogHazard_Objective(trial: optuna.trial.Trial):

    # Instantiate global Best_Model
    global _best_model

    # Suggest different learning rates
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-2)

    # Instantiate the LogisticHazard model
    LogHazard_Model = LogisticHazard(
        net = Define_Model(trial)
        , optimizer = tt.optim.Adam(lr = lr)
        , duration_index = Label_Transform.cuts
        , device = Device
    )

    # Train the model
    LogHazard_Model_log = LogHazard_Model.fit(
        input = x_train
        , target = y_train
        , batch_size = 40
        , epochs = 200
        , callbacks = [
            tt.callbacks.EarlyStopping()
            , Callback_Prune(trial)
        ]
        , verbose = False
        , val_data = (x_valid, y_valid)
    )

    # Store Model
    _best_model = LogHazard_Model

    # Store the minimum validation loss
    Val_Loss_Min = min(LogHazard_Model.val_metrics.scores['loss']['score'])

    # Return the validation loss
    return Val_Loss_Min

# TODO: Find cleaner way of storing best trained model during study

# Use SQLAlchemy to instantiate a RDB to store results
Study_DB = create_engine('sqlite:///Survival Analysis Studies/20200827_LogHazard_Study.db')

# Define callback to save the best_model
def BestModelCallback(study: optuna.study.Study, trial: optuna.trial.Trial):
    global Best_Model
    # Boolean check to see if current trial is the best performing
    if study.best_trial.number == trial.number:
        # Save the current model if best performing
        _best_model.save_net(
            path = 'Survival Analysis Studies//20200827_LogHazard_Model_{}.sav'.format(trial.number)
        )
        Best_Model = _best_model

# Run the optimization | no pruning since model.fit function bypasses individual steps
if __name__ == "__main__":
    # Instantiate the study
    LogHazard_Study = optuna.create_study(
        study_name = 'LogisticHazard'
        , direction = 'minimize'
        , sampler = optuna.samplers.TPESampler(
            seed = Random_Seed
            , consider_endpoints=True
        )
        , pruner = optuna.pruners.MedianPruner(
            n_startup_trials = 20
            , n_warmup_steps = 20
            , interval_steps = 5
        )
        , storage = 'sqlite:///Survival Analysis Studies/20200827_LogHazard_Study.db'
        , load_if_exists = True
    )
    # Start the optimization
    LogHazard_Study.optimize(
        LogHazard_Objective
        , n_trials = 50
        , n_jobs = Cores
        , callbacks = [BestModelCallback]
    )

    # Save best generated model
    LogHazard_Model_Best = Best_Model
    # LogHazard_Model_best_net = LogHazard_Study.user_attrs['best_model_net']
    # LogHazard_Model_best_weights = LogHazard_Study.user_attrs['best_model_weights']

    # Store the pruned and complete trials
    pruned_trials = [t for t in LogHazard_Study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in LogHazard_Study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    # Print statistics
    print("Study statistics: ")
    print("  Number of finished trials: ", len(LogHazard_Study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    # Store best_trial information and print it
    Best_Trial = LogHazard_Study.best_trial
    print("Best trial:")
    print("  Value: ", Best_Trial.value)
    print("  Params: ")
    for key, value in Best_Trial.params.items():
        print("    {}: {}".format(key, value))

# Best Valid_Loss = 3.807277

# Pickle the LogHazard_Study | no longer necessary as Study is stored in Sqlite3 DB
# pickle.dump(
#     obj = LogHazard_Study
#     , file = open('Survival Analysis Studies//20200824_LogHazard_Study.sav', 'wb')
# )


# <editor-fold desc="Access LogHazard_Study.db and evaluate key tables">

# Connect to db
SQLite_DB = create_engine('sqlite:///Survival Analysis Studies/20200827_LogHazard_Study.db')

# Instantiate MetaData object
DB_MetaData = MetaData(bind = SQLite_DB)

# Automap the existing tables
DB_MetaData.reflect()

# Create factory for session objects bound to db
Session = sessionmaker(bind = SQLite_DB)

# Create session object
session = Session()

# Display tables
print(DB_MetaData.tables.keys())

# Convert Trials to schema table
Trials = Table(
    'trials'
    , DB_MetaData
    , autoload = True
    , autoload_with = SQLite_DB
)

# Convert Trial_Params to schema table
Trial_Params = Table(
    'trial_params'
    , DB_MetaData
    , autoload = True
    , autoload_with = SQLite_DB
)

# Convert Trials into dataframe
Trials_df = pd.read_sql(
    sql = session.query(Trials).statement
    , con = session.bind
)

# </editor-fold>

# TESTING SECTION | Commenting out

# </editor-fold>