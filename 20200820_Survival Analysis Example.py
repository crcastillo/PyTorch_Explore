"""
    PyTorch Explore - Survival Analysis Example

    Objective: Construct various Survival Models within PyTorch framework and see if it can be stood up on GPU.

    Initial Build: 8/20/2020

    Notes:
    -
"""

# <editor-fold desc="Import relevant modules, load dataset, split Test/Train/Valid">

import torch
from torch.autograd import Variable
import torch.optim as optim
from pycox.models import LogisticHazard
import torchtuples as tt

import numpy as np
import pandas as pd
from sksurv.datasets import load_whas500

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from matplotlib import pyplot as plt
import plotly
from datetime import datetime
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
Label_Transform = LogisticHazard.label_transform(Target_DF_Convert(Y_Train_Split).duration.max().astype(int))
# Label_Transform = LogisticHazard.label_transform(20)

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

# Plot the train vs validation loss
LogHazard_Model_log.plot()

# Create survival curve predictions
x_test_pred = LogHazard_Model.predict_surv_df(input = x_test)


# </editor-fold>



# </editor-fold>