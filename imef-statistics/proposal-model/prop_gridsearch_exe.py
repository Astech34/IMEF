# imports
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import numpy as np
import pandas as pd
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import os
import json
from tqdm.auto import tqdm

import prop_utils as utils
import prop_data_prep as pdp
import prop_nn_models as pnn
import prop_nn_functions as pnf
import prop_MCDO_functions as MCDO

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# load data
# df_ds = pdp.df_ds  # mms test data (from Andy)
df_ds = pd.read_pickle("prop_models/complete_training_df")  # complete set (2019-2022)

# set predictors and target
# predictors = ["SYM/H_INDEX_nT", "X", "Y", "Z"]  # predictors, features
# target = ["EX", "EY", "EZ"]  # targets, outputs

predictors = [
    "OMNI_IMF",
    "OMNI_Vx",
    "OMNI_Vy",
    "OMNI_Vz",
    "OMNI_SYM_H",
    "L",
    "MLAT",
    "MLT",
]  # predictors, features
target = ["EX", "EY", "EZ"]  # targets, outputs
# target = ["EZ"]  # targets, outputs


# # hyperparameter configuration
# CONFIG = {
#     "hidden_size": [1000, 100, 15],  # no. of nuerons in hidden layer ;experiment
#     "learning_rate": 1e-5,  # learning rate,
#     "seq_len": 60,  # sequence length
#     "bsize": 32,  # batch size
#     "dropout_prob": 0.2,  # dropout probability
#     "num_epochs": 100,  # number of epochs
#     "patience": 20,  # stop-loss function patience
#     "num_features": len(predictors),  # no. of input features/variables
#     "output_size": len(target),  # predicting n vars where n = len(target) array
# }

seq_len = 60
bsize=32
dropout_prob=0.2
num_epochs = 1000
patience = 20
num_features = len(predictors)
output_size = len(target)

# hidden_size_arr = [[2500, 1000, 500, 100, 15],
#                        [1000, 500, 100, 15],
#                        [1000, 100, 10]]

hidden_size_arr = [[1500, 1000, 500, 100, 15], 
                   [1000, 500, 100, 15], 
                   [1000, 100, 10]]

learning_rate_arr = [1e-4, 1e-5, 1e-6]

# hidden_size_arr = [[100,50, 25], [50, 25,5]]
# learning_rate_arr = [1e-2,1e-4]


# main
def main():
    # MODEL SETUP

    # set a fixed value for the hash seed
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

    # (!) when running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # define target and predictions variables
    # x_train : feature vector (inputs) for training dataset
    # y_train : target vector (outputs) for training dataset
    # x_test : feature vector for testing dataset; used to evaluate models performace
    # y_test : test vector for testing dataset

    # during training, algorithm learns from x,y_train, adjusting to minimize error.
    # performance is evaluated by x,y_test.

    # split dataset into predictors (x) and target (y)
    x = df_ds[predictors]
    # x = df_ds[predictors].values # 5/10/24

    if len(target) > 1:
        y = df_ds[target]
        # y = df_ds[target].values  # 5/10/24
        # unsure if need to add .values, results are a little different
    else:
        y = df_ds[target].values.reshape(-1, 1)  # for one target

    # scaling the dataset
    # - (?) scale y?
    # - (!) consider min/max scaler (see that one article)
    x_scaler = StandardScaler()
    x_scaler.fit(x.values)
    x_scaled = x_scaler.transform(x)
    y_scaler = StandardScaler()

    # adjusting for single or multiple targets
    if len(target) > 1:
        y_scaler.fit(y.values)
        y_scaled = y_scaler.transform(y)
    else:
        y_scaler.fit(y)
        y_scaled = y_scaler.transform(y)

    # write as 3D arrays
    x_scaled3d, y_scaled3d = pdp.prepare_3Darrs(
        x_scaled, y_scaled, lag=seq_len, delay=1, next_steps=1
    )

    # define train and test sets
    #   - (!) always shuffling to best avoid cross-sampling
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled3d, y_scaled3d, test_size=0.3, random_state=seed, shuffle=True
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_test, y_test, test_size=0.5, random_state=seed, shuffle=True
    )  # test/validation set

    # covert data to torch tensors
    x_train_tensor = torch.Tensor(x_train)
    y_train_tensor = torch.Tensor(y_train)
    x_test_tensor = torch.Tensor(x_test)
    y_test_tensor = torch.Tensor(y_test)
    x_val_tensor = torch.Tensor(x_val)
    y_val_tensor = torch.Tensor(y_val)

    # text prompt gridsearch
    print('STARTING GRIDSEARCH....')

    # results to be saved in txt file
    results = []

    # model number marker
    model_number = 0

    for hidden_size in tqdm(hidden_size_arr, desc=f"Loop HSIZE"):
        for learning_rate in tqdm(learning_rate_arr, desc=f"Loop LRATE"):

            # set model number
            model_number += 1

            # skip first pass because we already did this in a separate run
            if hidden_size == hidden_size_arr[0] and learning_rate == learning_rate_arr[0]:
                continue

            # terminal check
            print(f"STARTING NEW TRAINING FOR MODEL #{model_number}")

            # initialize the model
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            model = pnn.FeedForwardNN_MCDO(
                num_features=num_features,
                seq_len=seq_len,
                hidden_sizes=hidden_size,
                output_size=output_size,
                dropout_prob=dropout_prob,
            )

            # define loss function and optimizer
            criterion = nn.MSELoss()  # mean squared error loss
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

            # check that model is properly initialized
            # print(f"Predictor(s): {predictors}, Target(s): {target}")
            # print(model)

            # training data loading
            train_data = TensorDataset(x_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_data, batch_size=bsize, shuffle=True)

            # validation data loading
            val_data = TensorDataset(x_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_data, batch_size=bsize, shuffle=True)

            # model shape check
            # print(x_train_tensor.shape, y_train_tensor.shape)
            # print(model(x_train_tensor).shape)

            # MODEL TRAINING

            test_loss = []  # list to hold test loss values over epoch
            val_loss = []  # list to hold validation loss values over epoch

            for epoch in range(num_epochs):
                model.train()  # turn on training mode
                loss_sum = 0
                v_loss_sum = 0

                # (1) training
                for inputs, targets in train_loader:
                    # forward pass
                    outputs = model(inputs)

                    # loss
                    loss = criterion(outputs, targets[:, 0, :])
                    loss_sum += loss.item()  # /len(inputs)

                    # back pass and optimization
                    optimizer.zero_grad()
                    loss.backward()  # compute gradients
                    optimizer.step()  # update weights

                # progress, computing test loss
                loss_sum = loss_sum / len(train_loader)
                test_loss.append(loss_sum)

                # (2) validation
                model.eval()
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        # forward pass
                        outputs = model(inputs)

                        # loss
                        v_loss = criterion(outputs, targets[:, 0, :])
                        v_loss_sum += v_loss.item()

                    # compute validation loss
                    v_loss_sum = v_loss_sum / len(val_loader)
                    val_loss.append(v_loss_sum)

                # (3) implement stop function
                # check if train loss has increase since last time
                latest_train_loss = val_loss[-2] if epoch > 1 else np.inf

                if v_loss_sum > latest_train_loss:
                    patience -= 1

                # reset to initial patience when model improves
                else:
                    patience = 20

                # break out if patience reached
                if patience == 0:
                    print(f"Patience reached at epoch {epoch}; stopping training...")
                    break

                # status check for every 50 epochs
                ep_lst = [1, 2, 4, 5, 6, 7, 8, 9]
                ep_lst_50 = [i * 50 for i in ep_lst]
                if epoch+1 in ep_lst_50:
                    print(f"Current Epoch: [{epoch + 1}/{num_epochs}], Test Loss: {loss_sum:.4f}, Validation Loss: {v_loss_sum:.4f}")

            # SAVES

            # save model locally
            print(f"SAVING MODEL #{model_number}")
            model_floc = "prop_models/"  # directory to save model
            model_fname = f"ANN_MODEL_N{model_number}"  # model name
            model_path = model_floc + model_fname
            torch.save(model.state_dict(), model_path)

            # compute and save metrics
            target_str = ""
            model.eval()
            with torch.no_grad():

                # get predicitons
                preds = model(x_test_tensor).detach().numpy() # predictions (scaled)
                preds_descaled = y_scaler.inverse_transform(preds) # predictions (descaled)
                y_test_descaled = y_scaler.inverse_transform(y_test[:, -1, :]) # actual data (descaled)

                for i in range(len(target)):
                    px = preds_descaled[:, i]
                    py = y_test_descaled[:, i]

                    # compute R, RMSE
                    rmse = MCDO.rmse(px, py)
                    ccoeff, pvalue = scipy.stats.pearsonr(px, py)
                    target_str = (
                        target_str + f"{target[i]} RMSE = {rmse:.4f},  R = {ccoeff:.4f} "
                    )

            # get model number and parameter string
            pstr = f"MODEL #{model_number}: LR = {learning_rate}, NUERONS = {hidden_size} | "
            full_string = pstr+target_str

            # print results
            print(f"RESULTS: {full_string}")

            # append to result list
            results.append(full_string)

    # at completion of search, save all metric results to .txt file
    with open('results.txt', 'w+') as f:
        # write elements of list
        for items in results:
            f.write('%s\n' %items)
            print(f"{pstr} .. File written successfully ..")
            print("TRAINING END.")

    # to load, use:
    # model.load_state_dict(torch.load(fname))

    # # save hyperparameter configuration
    # with open(model_path + "_config.json", "w") as f:
    #     json.dump(CONFIG, f)

    # print("Done. Saving loss plot...")
    # # plot test and validation loss
    # fig, ax = plt.subplots(1)
    # ax.plot(np.arange(epoch + 1), test_loss, label="test loss")
    # ax.plot(np.arange(epoch + 1), val_loss, label="validation loss")
    # ax.set_xlabel("epoch")
    # ax.set_ylabel("loss")
    # ax.legend()
    # # save figure
    # utils.save_fig_loc(fname="loss_plot_EXYZ", path="prop_figures/", override=True)


if __name__ == "__main__":
    main()
