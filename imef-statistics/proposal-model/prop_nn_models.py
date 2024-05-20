import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import os

# basic mulit-layer feed-forward nueral network 

class FeedForwardNN(nn.Module):
    def __init__(self, num_features, seq_len, hidden_sizes, output_size):
        if not isinstance(hidden_sizes, list): hidden_sizes = [hidden_sizes]
        """
        input_size is int
        hidden_sizes is list (e.g. [60] or [60,30])
        output_size is int
        """
        super(FeedForwardNN, self).__init__()
        """
        self.fc1 = nn.Linear(input_size, hidden_size)   # first fully connected layer
        self.relu = nn.ReLU()                           # ReLU activation function (!) leaky relu?
        self.fc2 = nn.Linear(hidden_size, output_size)  # final fully connected layer
        """

        layers = []
        prev_size = num_features * seq_len
        for next_size in hidden_sizes:
            layers.append( nn.Linear(prev_size, next_size) )
            layers.append( nn.ReLU() )
            prev_size = next_size
        layers.append( nn.Linear(hidden_sizes[-1], output_size) )
        self.layers = nn.Sequential(*layers)

        self.num_features = num_features
        self.seq_len = seq_len
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
    def forward(self, x):
        # So when modeling time series with feed-forward NN's, it's important to note that there's a neuron
        # for each time step AND FEATURE; so if you have 60 time steps and 2 features, then you have 120 numbers
        # to give as input; so the NN see the time steps and features *at the same time*
        # we can express this idea by taking our 3d array (with original shape [num_pts, seq_len, num_feats])
        # and unfold into 2d (with new shape [num_pts, seq_len * num_feats])
        x = x.reshape( (x.shape[0], self.seq_len * self.num_features ) )
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.fc2(out)
        """
        return self.layers(x)
    


class FeedForwardNN_MCDO(nn.Module):
    def __init__(self, num_features, seq_len, hidden_sizes, output_size, dropout_prob):
        
        if not isinstance(hidden_sizes, list): hidden_sizes = [hidden_sizes]
        super(FeedForwardNN_MCDO, self).__init__()

        layers = []
        prev_size = num_features * seq_len
        for next_size in hidden_sizes:
            layers.append( nn.Linear(prev_size, next_size) )
            layers.append( nn.ReLU() )
            prev_size = next_size
            layers.append(nn.Dropout(p=dropout_prob)) # (*) MC DROPOUT, may need to be done for every hidden size (in or our for loop?)
        layers.append( nn.Linear(hidden_sizes[-1], output_size) )
        self.layers = nn.Sequential(*layers)

        self.num_features = num_features
        self.seq_len = seq_len
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
    def forward(self, x):
        #  When modeling time series with feed-forward NN's, it's important to note that there's a neuron
        # for each time step AND FEATURE; so if you have 60 time steps and 2 features, then you have 120 numbers
        # to give as input; so the NN see the time steps and features *at the same time*
        # we can express this idea by taking our 3d array (with original shape [num_pts, seq_len, num_feats])
        # and unfold into 2d (with new shape [num_pts, seq_len * num_feats])
        x = x.reshape( (x.shape[0], self.seq_len * self.num_features ) )
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.fc2(out)
        """
        return self.layers(x)