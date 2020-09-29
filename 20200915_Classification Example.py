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
import os

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

# # Define training data class
# class trainData(torch.utils.data.Dataset):
#     def __init__(self, x_train, y_train):
#         self.x_train = x_train
#         self.y_train = y_train
#     def __getitem__(self, index):
#         return self.x_train[index], self.y_train[index]
#     def __len__(self):
#         return len(self.x_train)

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
NN_Lit_Model_trained.to('cpu')
NN_Lit_Model_trained(torch.randn(42))

# Score Test data
Pred = np.array([])
Actual = np.array([])
for _inputs, _targets in test_dl:
    # Load inputs to GPU
    _inputs_t = _inputs.to('cpu')
    # Compute model output
    _yhat = NN_Lit_Model_trained(_inputs_t)
    # Append predictions and targets
    Pred = np.append(Pred, _yhat.cpu().data.numpy())
    Actual = np.append(Actual, _targets.cpu().data.numpy())
# Determine AUC and Average Precision Score
print('AUC = {:.4f}'.format(roc_auc_score(y_true=Actual, y_score=Pred)))
print('Average Precision Score = {:.4f}'.format(average_precision_score(y_true=Actual, y_score=Pred)))

# AUC = 0.9290
# Average Precision Score = 0.6131

# </editor-fold>

# <editor-fold desc="Use Optuna to hypertune the NeuralNet_Lit model and create predictions">

# Construct the neural net class
class NNet(torch.nn.Module):
    def __init__(self, trial):
        super(NNet, self).__init__()
        self.layers = []
        self.dropouts = []

        # Optimize the number of hidden layers, hidden units, and dropout percentages
        n_hidden_layers = trial.suggest_int('n_hidden_layers', 1, 3)
        input_features = x_train.shape[1]
        output_features_max = input_features * 2 - 1

        # Append hidden layers and dropouts
        for _i in range(n_hidden_layers):
            # Store potential out_features
            output_features = trial.suggest_int('n_units_l{}'.format(_i), low=4, high=output_features_max, log=True)
            # Append a hidden layer
            self.layers.append(torch.nn.Linear(in_features=input_features, out_features=output_features))
            # Suggest a proportion for Dropout
            p = trial.suggest_uniform("dropout_l{}".format(_i), 0.05, 0.5)
            # Dropout layer, which randomly zeroes some proportion of the elements
            self.dropouts.append(torch.nn.Dropout(p=p))
            # Set input_features accordingly so any new layers take on the input dimension from the last output dimension
            input_features = output_features
        # Append final layer
        self.layers.append(torch.nn.Linear(in_features=input_features, out_features=1))
        # Assigning the layers as class variables (PyTorch requirement)
        for idx, layer in enumerate(self.layers):
            setattr(self, "fc{}".format(idx), layer)
        # Assigning the dropouts as class variables (PyTorch requirement)
        for idx, dropout in enumerate(self.dropouts):
            setattr(self, "drop{}".format(idx), dropout)
    # Define the forward pass
    def forward(self, x):
        for layer, dropout in zip(self.layers, self.dropouts):
            x = torch.relu(layer(x))
            x = dropout(x)
        return torch.sigmoid(self.layers[-1](x))

# Create the PyTorch Lightning class
class NNet_Lightning(pl.LightningModule):
    def __init__(self, trial):
        super(NNet_Lightning, self).__init__()
        self.model = NNet(trial)
        # Suggest different learning rates
        self.lr = trial.suggest_loguniform('lr', 1e-6, 1e-1)
    # Define the forward pass output
    def forward(self, x):
        return self.model(x)
    # Define optimizer
    def configure_optimizers(self):
        return torch.optim.SGD(params=self.model.parameters(),lr=self.lr)
    # Define the training step
    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = torch.nn.functional.binary_cross_entropy(input=self.forward(x), target=y)
        return {'loss': loss}
    # Define the validation step
    def validation_step(self, batch, batch_nb):
        x, y = batch
        loss = torch.nn.functional.binary_cross_entropy(input=self.forward(x), target=y)
        return {'batch_val_loss': loss}
    # Define the validation post epoch end step
    def validation_epoch_end(self, outputs):
        avg_loss = sum(x["batch_val_loss"] for x in outputs) / len(outputs)
        return {"log": {"avg_val_loss": avg_loss}}

# Create PyTorch Lightning metric callback
class MetricsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []
    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

# Define objective function to optimize within Optuna study
def NNet_Objective(trial):
    # Create trial model checkpoint
    TrialCheckpointCallback = pl.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            'Classifier Studies/'
            ,'trial_{}'.format(trial.number)
            , '_{epoch}'
        )
        , monitor='avg_val_loss'
    )
    # Instantiate the MetricsCallback
    Metrics_Callback = MetricsCallback()

    # Instantiate the PyTorch Lightning Trainer
    PL_Trainer = pl.Trainer(
        logger=False
        , checkpoint_callback=TrialCheckpointCallback
        , max_epochs=250
        , gpus=[Device]
        , callbacks=[Metrics_Callback]
        , early_stop_callback=optuna.integration.PyTorchLightningPruningCallback(trial, monitor='avg_val_loss')
        , progress_bar_refresh_rate=0
    )

    # Instantiate the PyTorch Lightning module
    Model = NNet_Lightning(trial)

    # Use Trainer to fit the model
    PL_Trainer.fit(
        model=Model
        , train_dataloader=train_dl
        , val_dataloaders=valid_dl
    )

    # Return the average validation loss
    return Metrics_Callback.metrics[-1]['avg_val_loss'].item()


# Use SQLAlchemy to instantiate a RDB to store results
Study_DB = create_engine('sqlite:///Classifier Studies/20200929_Classifier_Study.db')

# Run the optimization | no pruning since model.fit function bypasses individual steps
if __name__ == "__main__":
    # Instantiate the study
    Classifier_Study = optuna.create_study(
        study_name = 'Classifier'
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
        , storage = 'sqlite:///Classifier Studies/20200930_Classifier_Study.db'
        , load_if_exists = True
    )
    # Start the optimization
    Classifier_Study.optimize(
        NNet_Objective
        , n_trials = 50
        , n_jobs = Cores
    )

    # Store the pruned and complete trials
    pruned_trials = [t for t in Classifier_Study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in Classifier_Study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    # Print statistics
    print("Study statistics: ")
    print("  Number of finished trials: ", len(Classifier_Study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    # Store best_trial information and print it
    Best_Trial = Classifier_Study.best_trial
    print("Best trial:")
    print("  Value: ", Best_Trial.value)
    print("  Params: ")
    for key, value in Best_Trial.params.items():
        print("    {}: {}".format(key, value))

# Best Avg Valid_Loss = ?

# </editor-fold>