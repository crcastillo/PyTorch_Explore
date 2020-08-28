"""
    PyTorch Explore - Survival Analysis Example

    Objective: Construct various Survival Models within PyTorch framework and see if it can be stood up on GPU.

    Initial Build: 8/20/2020

    Notes:
    - Use the save_model_weights method for LogisticHazard to save the neural net model; preferable to save_net
    - Cannot append trial.model with set_user_attrs since it's not JSON serializable
    - Saving best current model with Callback
"""

# <editor-fold desc="Import relevant modules, load dataset, split Test/Train/Valid">

import torch
from torch.autograd import Variable
import torch.optim as optim
import torchtuples as tt
from pycox.models import LogisticHazard
from pycox.evaluation import EvalSurv

import numpy as np
import pandas as pd
from sksurv.datasets import load_whas500

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sqlalchemy import create_engine

# from matplotlib import pyplot as plt
# import plotly
# from datetime import datetime
import pickle

import optuna
import multiprocessing


# Set parameters
Random_Seed = 123
Test_Proportion = 0.2
Cores = np.int(multiprocessing.cpu_count() / 2)
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import dataset
Data_X, Data_Y = load_whas500()

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

# Create function to convert structured array into DataFrame
def Target_DF_Convert(_array: np.ndarray):
    _df = pd.DataFrame(
        data = np.array(_array.tolist())
        , columns = ['event', 'duration']
    )
    return _df

# Instantiate label_transform with number of cuts equal to max duration
# Label_Transform = LogisticHazard.label_transform(20)
# Label_Transform = LogisticHazard.label_transform(Target_DF_Convert(Y_Train_Split).duration.max().astype(int))
Label_Transform = LogisticHazard.label_transform(
    cuts = np.array(
        range(
            0 # Target_DF_Convert(Y_Train_Split).duration.min().astype(int)
            , Target_DF_Convert(Y_Train_Split).duration.max().astype(int)
        )
        , dtype = float
    )
)

# Build transformer to alter
get_target = lambda df: (df['duration'].values, df['event'].values)

# Create targets for training and validation
y_train = Label_Transform.fit_transform(*get_target(Target_DF_Convert(Y_Train_Split)))
y_valid = Label_Transform.transform(*get_target(Target_DF_Convert(Y_Valid_Split)))

# Apply StandardScaler for feature matrix
Train_Transformer = ColumnTransformer(
    transformers = [
        ('StdScaler', StandardScaler(), ['age', 'bmi', 'diasbp', 'hr', 'los', 'sysbp'])
    ]
    , remainder = 'passthrough'
)

# Fit and transform feature matrices
x_train = Train_Transformer.fit_transform(X = X_Train_Split).astype('float32')
x_valid = Train_Transformer.transform(X = X_Valid_Split).astype('float32')
x_test = Train_Transformer.transform(X = X_Test).astype('float32')

# </editor-fold>

# <editor-fold desc="Create NeuralNet class and instantiate the model">

# Inherit Sequential NeuralNet module
NeuralNet = torch.nn.modules.Sequential(

    # Input layer and first hidden layer
    torch.nn.modules.linear.Linear(
        in_features = X_Train_Split.shape[1]
        , out_features = 42
    )
    # Activation function
    , torch.nn.modules.activation.ReLU()
    # Normalize the output for each batch
    , torch.nn.modules.batchnorm.BatchNorm1d(num_features = 42)
    # Dropout layer, which randomly zeroes some proportion of the elements
    , torch.nn.modules.dropout.Dropout(p = 0.1)

    # Second hidden layer
    , torch.nn.modules.linear.Linear(
        in_features=42
        , out_features=42
    )
    # Activation function
    , torch.nn.modules.activation.ReLU()
    # Normalize the output for each batch
    , torch.nn.modules.batchnorm.BatchNorm1d(num_features = 42)
    # Dropout layer
    , torch.nn.modules.dropout.Dropout(p = 0.1)

    # Output
    , torch.nn.modules.linear.Linear(
        in_features = 42
        , out_features = Label_Transform.out_features
    )
)

# Instantiate the LogisticHazard model
LogHazard_Model = LogisticHazard(
    net = NeuralNet
    , optimizer = tt.optim.Adam(lr = 0.01)
    , duration_index = Label_Transform.cuts
    , device = Device
)

# </editor-fold>

# <editor-fold desc="Fit the LogisticHazard model and create predictions">

# Train the model
LogHazard_Model_log = LogHazard_Model.fit(
    input = x_train
    , target = y_train
    , batch_size = 40
    , epochs = 200
    , callbacks = [tt.callbacks.EarlyStopping()]
    , verbose = True
    , val_data = (x_valid, y_valid)
)

# Print the minimum validation loss value
print(LogHazard_Model_log.to_pandas().val_loss.min())

# Plot the train vs validation loss
LogHazard_Model_log.plot()

# Create survival curve predictions
x_test_pred = LogHazard_Model.predict_surv_df(input = x_test)

# Create evaluation
test_eval = EvalSurv(
    surv = x_test_pred
    , durations = get_target(Target_DF_Convert(Y_Test))[0]
    , events = get_target(Target_DF_Convert(Y_Test))[1]
    , censor_surv = 'km'
)

# Print the concodance index (Antolini method)
print(test_eval.concordance_td())

# </editor-fold>

# <editor-fold desc="Use Optuna to hypertune the LogisticHazard model and create predictions">

# Create Callback class to attempt end of epoch trial.prune
class Callback_Prune(tt.callbacks._ActionOnBestMetric):
    def __init__(self, trial: optuna.trial.Trial, metric='loss', dataset='val', get_score=None):
        super().__init__(metric, dataset, get_score)
        self._trial = trial
        self._epoch = -1

    def on_epoch_end(self):
        # Advance _epoch by 1
        self._epoch += 1
        # Retrieve the current val score
        score = self.get_score()
        # Report score to trial
        self._trial.report(value = score, step = self._epoch)
        # Run pruning boolean
        if self._trial.should_prune():
            message = 'Trial was pruned at iteration {}.'.format(self._epoch)
            raise optuna.TrialPruned(message)

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
        , callbacks = [tt.callbacks.EarlyStopping(), Callback_Prune(trial)]
        , verbose = False
        , val_data = (x_valid, y_valid)
    )

    # Store Model
    _best_model = LogHazard_Model

    # Store the minimum validation loss
    Val_Loss_Min = min(LogHazard_Model.val_metrics.scores['loss']['score'])

    # Return the validation loss
    return Val_Loss_Min



# TODO: Figure out how to prevent the weight_checkpoint saving from the Callback_Prune
# TODO: Find cleaner way of storing best trained model during study

# Use SQLAlchemy to instantiate a RDB to store results
Study_DB = create_engine('sqlite:///Survival Analysis Studies/20200827_LogHazard_Study.db')

# Define callback to save the best_model
def BestModelCallback(study: optuna.study.Study, trial: optuna.trial.Trial):
    global Best_Model
    # Boolean check to see if current trial is the best performing
    if study.best_trial.number == trial.number:
        # Append an attribute to the study and save the current model if best performing
        # Best_Model.save_model_weights(
        #     path = 'Survival Analysis Studies//20200825_LogHazard_Model_{}.sav'.format(trial.number)
        # )
        _best_model.save_net(
            path = 'Survival Analysis Studies//20200827_LogHazard_Model_{}.sav'.format(trial.number)
        )
        Best_Model = _best_model


# Run the optimization | no pruning since model.fit function bypasses individual steps
if __name__ == "__main__":
    # Instantiate the study
    LogHazard_Study = optuna.create_study(
        direction = 'minimize'
        , sampler = optuna.samplers.TPESampler(seed = Random_Seed)
        , pruner = optuna.pruners.MedianPruner(
            n_startup_trials = 10
            , n_warmup_steps = 1
            , interval_steps = 1
        )
        , storage = 'sqlite:///Survival Analysis Studies/20200827_LogHazard_Study.db'
    )
    # Start the optimization
    LogHazard_Study.optimize(
        LogHazard_Objective
        , n_trials = 20
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

# Pickle the LogHazard_Study
pickle.dump(
    obj = LogHazard_Study
    , file = open('Survival Analysis Studies//20200824_LogHazard_Study.sav', 'wb')
)


# TESTING SECTION

TEST_Model = LogisticHazard(
    net = NeuralNet
    , optimizer = tt.optim.Adam(lr = 0.01)
    , duration_index = Label_Transform.cuts
    , device = Device
)

TEST_Model.load_net(
    path = 'Survival Analysis Studies//20200825_LogHazard_Model_45.sav'
)

TEST_Model_pred = TEST_Model.predict_surv_df(input = x_test)
Best_Model_pred = Best_Model.predict_surv_df(input = x_test)

TEST_test_eval = EvalSurv(
    surv = TEST_Model_pred
    , durations = get_target(Target_DF_Convert(Y_Test))[0]
    , events = get_target(Target_DF_Convert(Y_Test))[1]
    , censor_surv = 'km'
)
Best_test_eval = EvalSurv(
    surv = Best_Model_pred
    , durations = get_target(Target_DF_Convert(Y_Test))[0]
    , events = get_target(Target_DF_Convert(Y_Test))[1]
    , censor_surv = 'km'
)


print(TEST_test_eval.concordance_td())
print(test_eval.concordance_td())
print(Best_test_eval.concordance_td())

# </editor-fold>