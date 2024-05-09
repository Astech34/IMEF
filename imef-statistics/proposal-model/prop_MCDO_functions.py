import matplotlib.pyplot as plt 
import scipy
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def rmse(y_test, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) between the test and predicted datasets.
    
    Args:
        y_test (array): Array of true target values, typically test set to be evaluated.
        y_pred (array): Array of predicted target values.
    
    Returns:
        RMSE value (float)
    """
    return np.sqrt(np.mean((y_test - y_pred) ** 2))


def enable_dropout(model):
    """
    Enable (turn on) dropout layers during predicitons/tests.
    Used for Monte-Carlo Dropout (MCDO)

    Args:
        model (model class): model to be evaluated
    """
    for mod in model.modules():
        if mod.__class__.__name__.startswith('Dropout'):
            mod.train()


# example
# iterate MC dropout over defined number of samples to get distribution of error
# num_passes = 100                        # number of monte-carlo passes to generate
# num_samples = len(model(x_test_tensor)) # number of samples in each target
# num_targets = output_size               # number of target variables in dataset

def get_MCDO_predictions(model, xtest, num_passes, num_samples, num_targets):
    """
    Run Monte-Carlo Dropout (MCDO) method and compute statistics.
        - MCDO estimates uncertainty by performing multiple forward passes (num_passes)
            with dropout enabled during inference, aggregating predictions, 
            and computing statistics to quantify uncertainty in prediction. 
        - Works by turning on/off diff. configurations of nuerons during forward passes
            to yield a predictive distribution over the mean. 
        - (!) Dropout layers of model MUST to train mode during evaluation.

    Args:
        model (class): NN model for evaluation
        xtest (pytorch tensor): x-test pytorch tensor for model evaluation
        num_passes (int): number of monte-carlo passes to generate
        num_samples (int): number of samples in each target
        num_targets (int):  number of target variables in dataset

    Returns:
        dropout predicitons and statistics (mean and standard dev.), all np arrays
    """

    dropout_predictions = np.empty((num_passes, num_samples, num_targets))

    # set model to evaluation mode
    model.eval() 

    # turn dropout ON during model evaluation
    enable_dropout(model) 

    with torch.no_grad():

        for i in range(num_passes):
            output = model(xtest) # shape (num_samples, num_targets)
            dropout_predictions[i] = output

    # compute mean, standard dev. of MCDO along passes (axis 0) for all samples in each target variable
    pred_mean = np.mean(dropout_predictions, axis=0) # shape (num_samples, num_targets)
    pred_stdev = np.var(dropout_predictions, axis=0)  # shape (num_samples, num_targets)

    if model.__class__.__name__.startswith('Dropout'):
            model.eval()

    return dropout_predictions, pred_mean, pred_stdev
