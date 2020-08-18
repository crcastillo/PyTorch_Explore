"""
    PyTorch Explore - Simple Linear Regression Example

    Objective: Construct a Linear Regression within PyTorch framework and see if it can be stood up on GPU.

    Initial Build: 8/10/2020

    Notes:
    - I need to figure out how to normalize the input matrix otherwise the model is very sensitive to learning rate
        and can quickly result in missing the local minima
"""

# <editor-fold desc="Import relevant modules">

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
from datetime import datetime

import optuna
import multiprocessing

# </editor-fold>

# <editor-fold desc="Load dataset and split into Train/Test">

# Load regression dataset (boston) from sklearn
X, Y = load_boston(return_X_y = True)

# Split Train/Test
X_Train, X_Test, Y_Train, Y_Test = train_test_split(
    X
    , Y
    , test_size = 0.20
    , random_state = 123
)

# Set the device
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.get_device_name(Device)

# Set the number of processing cores and divide by 2
Cores = np.int(multiprocessing.cpu_count() / 2)

# </editor-fold>

# <editor-fold desc="Build LinearRegression Class and instantiate the model">

# Inherit the basic NeuralNetwork module
class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
    # Define the forward pass output
    def forward(self, X):
        out = self.linear(X)
        return out

# Instantiate the model
LinearModel = LinearRegression(
    input_size = X_Train.shape[1]
    , output_size = 1
)

# If GPU available then instantiate model on GPU
LinearModel.to(Device)

# </editor-fold>

# <editor-fold desc="Define model parameters and train">

# Define model parameters
epochs = 10000
learning_rate = 0.0000001
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(
    params = LinearModel.parameters()
    , lr = learning_rate
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer = optimizer
    , mode = 'min'
    , patience = 10
    , cooldown = 5
    , threshold = 1
)

# Store the initial start time
StartTime = datetime.now()

# Train the model
Train_MSE_Loss = [] # Instantiate MSE_Loss for model training
for _epoch in range(epochs):

    # Convert inputs and labels to Variable make sure to convert to float first
    _inputs = Variable(torch.from_numpy(X_Train).float().to(Device))
    _labels = Variable(torch.from_numpy(Y_Train).float().to(Device))

    # Store output from LinearModel as a function of inputs
    _outputs = LinearModel(_inputs)

    # Store the loss
    _loss = criterion(_outputs, _labels.unsqueeze_(1))
    Train_MSE_Loss += [_loss]

    # Clear gradient buffer from previous epochs, don't want to accumulate gradients
    optimizer.zero_grad()

    # Store gradient with respect to parameters
    _loss.backward()

    # Reduce learning rate if learning stagnates | TEST
    scheduler.step(_loss)

    # Update parameters
    optimizer.step()

    # Print it
    print('epoch {}, loss {}'.format(_epoch, _loss.item()))

# </editor-fold>

# <editor-fold desc="Test the model and plot predictions">

# Create predictions
with torch.no_grad():
    Pred_Y = LinearModel(Variable(torch.from_numpy(X_Test).float().to(Device))).cpu().data.numpy()

# Store Mean Squared Error
Test_MSE = np.square(np.subtract(Pred_Y.reshape(Pred_Y.shape[0], ), Y_Test)).mean()
print(Test_MSE)

# Display the runtime
print(datetime.now() - StartTime)

# Create Y_Test sorting index
Y_Test_idx = Y_Test.argsort()

# Plot performance figure
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(14, 10))
plt.plot(Y_Test[Y_Test_idx], 'go', label = 'Actual', alpha = 0.5)
plt.plot(Pred_Y[Y_Test_idx], '--', label = 'Predict', alpha = 0.5)
plt.legend(loc = 'best')
plt.show()


# TEST = torch.rand(10, 20)
# TEST_normal_dim1 = F.normalize(input = TEST, p = 2, dim = 1) # Vector wise normalization
# TEST_normal_dim0 = F.normalize(input = TEST, p = 2, dim = 0) # Column wise normalization


# </editor-fold>

# <editor-fold desc="Build LinearRegression Function and Train the model with Op">

# Define LinearRegression module
# def LinearRegression_Model(trial):

# Create objective
def LinearRegression_Objective(trial):

    # Instantiate the model and send to GPU if available
    LinearModel = LinearRegression(
        input_size = X_Train.shape[1]
        , output_size = 1
    ).to(Device)

    # Generate the optimizer search space
    optimizer_name = trial.suggest_categorical('optimizer', ['SGD', 'RMSprop', 'Adam'])
    # Generate the learning rate search space
    learning_rate = trial.suggest_loguniform('lr', 1e-9, 1e-3)
    # Generate the momentum search space
    if optimizer_name in ['SGD', 'RMSprop']:
        momentum = trial.suggest_uniform('momentum', 0.4, 0.99)

    # Define the optimizer differently for non-Adam vs Adam
    if optimizer_name in ['SGD', 'RMSprop']:
        optimizer = getattr(optim, optimizer_name)(
            params = LinearModel.parameters()
            , lr = learning_rate
            , momentum = momentum
        )
    else:
        optimizer = getattr(optim, optimizer_name)(
            params = LinearModel.parameters()
            , lr = learning_rate
        )

    # Instantiate MSELoss as the optimizer criterion
    criterion = torch.nn.MSELoss()

    # Train the model
    Train_MSE_Loss = []  # Instantiate MSE_Loss for model training
    for _epoch in range(epochs):

        # Convert inputs and labels to Variable make sure to convert to float first
        _inputs = Variable(torch.from_numpy(X_Train).float().to(Device))
        _labels = Variable(torch.from_numpy(Y_Train).float().to(Device))

        # Store output from LinearModel as a function of inputs
        _outputs = LinearModel(_inputs)

        # Store the loss
        _loss = criterion(_outputs, _labels.unsqueeze_(1))
        Train_MSE_Loss += [_loss]

        # Clear gradient buffer from previous epochs, don't want to accumulate gradients
        optimizer.zero_grad()

        # Store gradient with respect to parameters
        _loss.backward()

        # Update parameters
        optimizer.step()

        # Create predictions
        with torch.no_grad():
            Pred_Y = LinearModel(Variable(torch.from_numpy(X_Test).float().to(Device))).cpu().data.numpy()

        # Store Mean Squared Error
        Test_MSE = np.square(np.subtract(Pred_Y.reshape(Pred_Y.shape[0], ), Y_Test)).mean()

        # Report progress
        trial.report(Test_MSE, _epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Return the model performance metric
    return Test_MSE


# Define model parameters
epochs = 10000

# Run the optimization
if __name__ == "__main__":
    # Instantiate the study
    LinearRegression_Study = optuna.create_study(
        direction = 'minimize'
        # , sampler = optuna.samplers.TPESampler
        , pruner = optuna.pruners.MedianPruner(
            n_startup_trials = 20
            , n_warmup_steps = 100
            , interval_steps = 10
        )
    )
    # Start the optimization
    LinearRegression_Study.optimize(
        LinearRegression_Objective
        , n_trials = 200
        , n_jobs = Cores
    )

    # Store the pruned and complete trials
    pruned_trials = [t for t in LinearRegression_Study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in LinearRegression_Study.trials if t.state == optuna.structs.TrialState.COMPLETE]

    # # Print statistics
    # print("Study statistics: ")
    # print("  Number of finished trials: ", len(LinearRegression_Study.trials))
    # print("  Number of pruned trials: ", len(pruned_trials))
    # print("  Number of complete trials: ", len(complete_trials))
    #
    # print("Best trial:")
    # Best_Trial = LinearRegression_Study.best_trial
    #
    # print("  Value: ", Best_Trial.value)
    #
    # print("  Params: ")
    # for key, value in Best_Trial.params.items():
    #     print("    {}: {}".format(key, value))

# </editor-fold>