# %%
###########
# IMPORTS #
###########

from pathlib import Path
from tabnanny import verbose

# Change this path to adapt to where you downloaded the data
DATA_PATH = Path("./geolifeclef-2022-lifeclef-2022-fgvc9")

################
### PACKAGES ###
################
import gc
from turtle import color
from sklearn.utils import shuffle
# del variables
gc.collect()

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim 
torch.cuda.empty_cache()

import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as utils
from torchvision import transforms    
from torch.optim import lr_scheduler as lrs


from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier

# Standard packages
import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle as pkl

# For evaluation and submission
from GLC.metrics import top_30_error_rate, predict_top_30_set, predict_top_k_set #,top_k_error_rate_from_sets
from GLC.submission import generate_submission_file
from sklearn.metrics import accuracy_score

# For data loading and visualization
from GLC.data_loading.common import load_patch
# from GLC.plotting import visualize_observation_patch
from GLC.data_loading.environmental_raster import PatchExtractor

# For time monitoring
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline

torch.manual_seed(42)
np.random.seed(42)




### DATA LOADING ###
# Load landcover metadata to use the patches
# df_suggested_landcover_alignment = pd.read_csv(DATA_PATH / "metadata" / "landcover_suggested_alignment.csv", sep=";")
# landcover_mapping = df_suggested_landcover_alignment["suggested_landcover_code"].values

# # Environmental features
# df_env = pd.read_csv("./enriched_df/df_features_coord_alt_land.csv", index_col="observation_id")
# df_env = (df_env-df_env.min())/(df_env.max()-df_env.min())   # min-max normalization
# # df_env = (df_env-df_env.mean())/df_env.std() # mean-std normalization


# # Species information
# df_species = pd.read_csv("./geolifeclef-2022-lifeclef-2022-fgvc9/metadata/species_details.csv", sep=";")#, index_col="species_id")
# species = df_species.GBIF_kingdom_name.values


# # Load train set of observations from France and USA and merge
# df_obs_fr = pd.read_csv(DATA_PATH / "observations" / "observations_fr_train.csv", sep=";", index_col="observation_id")
# df_obs_us = pd.read_csv(DATA_PATH / "observations" / "observations_us_train.csv", sep=";", index_col="observation_id")
# df_obs = pd.concat((df_obs_fr, df_obs_us))


# # Extract species kingdom
# df_obs['kingdom'] = df_obs.apply(lambda x: species[x.species_id], axis=1)


# # Extract training and validation subsets as np arrays
# obs_id_train = df_obs.index[df_obs["subset"] == "train"].values
# obs_id_val = df_obs.index[df_obs["subset"] == "val"].values

# df_train =  df_obs.loc[obs_id_train].reset_index(drop=False)
# df_val =  df_obs.loc[obs_id_val].reset_index(drop=False)

# # Same with test set of observations
# df_obs_fr_test = pd.read_csv(DATA_PATH / "observations" / "observations_fr_test.csv", sep=";", index_col="observation_id")
# df_obs_us_test = pd.read_csv(DATA_PATH / "observations" / "observations_us_test.csv", sep=";", index_col="observation_id")
# df_obs_test = pd.concat((df_obs_fr_test, df_obs_us_test))

# # Extract observaions as np array
# obs_id_test = df_obs_test.index.values
# df_test =  df_obs_test.loc[obs_id_test].reset_index(drop=False)

# %%
# Load the environmental vectors
# df_features = pd.read_csv("./enriched_df/df_features_coord_alt_land.csv", sep=",", index_col="observation_id")

# for c in ["cnn_"+str(i) for i in range(1024)]:
#     df_features[c] = np.float16(0)

# # Fill nan values
# df_features.fillna(np.finfo(np.float32).min, inplace=True)

# # Display the result
# # display(df_features.head(3))


# # Load landcover metadata to use the patches
# df_landcover_labels = pd.read_csv(DATA_PATH / "metadata" / "landcover_original_labels.csv", sep=";")
# df_suggested_landcover_alignment = pd.read_csv(DATA_PATH / "metadata" / "landcover_suggested_alignment.csv", sep=";")
# landcover_mapping = df_suggested_landcover_alignment["suggested_landcover_code"].values

# display(df_landcover_labels.head(2))
# display(df_suggested_landcover_alignment.head(2))

# %% [markdown]
# Enrich with pd.apply and monitor with tqdm

# %%
# class ImageDataTest(Dataset):
#     def __init__(self, df_env, data_path, load, landcover_mapping):
#         super().__init__()
#         self.df_env = df_env
#         self.data_path = data_path
#         self.load = load
#         self.landcover_mapping = landcover_mapping

#     def __len__(self):
#         return self.df_env.shape[0]
    
#     def __getitem__(self, idx):
#         obs_id = self.df_env.iloc[idx].name
        
#         patch = self.load(obs_id, self.data_path, landcover_mapping=self.landcover_mapping)
#         patch = [patch[0].astype(np.float32)/255,
#                  np.reshape(patch[1], (patch[1].shape[0], patch[1].shape[1],1)).astype(np.float32)/255,
#                 #   np.reshape(patch[2], (patch[2].shape[0], patch[2].shape[1],1)),
#                    np.reshape(patch[3], (patch[3].shape[0], patch[3].shape[1],1)).astype(np.float32)/33]
#         patch = torch.FloatTensor(np.concatenate(patch, axis=2))
#         patch = torch.movedim(patch, 2, 0)

#         return patch, obs_id


# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.cnn = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        
#         self.cnn.conv1 = nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

#         self.cnn.fc = nn.Sequential(
#                                     nn.BatchNorm1d(num_features=512+31),
#                                     nn.Linear(in_features=512 + 31 , out_features=4096, bias=True),
#                                     nn.ReLU(),
#                                     nn.Dropout(p=0.5),
#                                     nn.Linear(in_features=4096, out_features=17037, bias=True)
#                                             )

#     def forward(self, patch):
#         x1 = self.cnn.conv1(patch)
#         x1 = self.cnn.bn1(x1)
#         x1 = self.cnn.relu(x1)
#         x1 = self.cnn.maxpool(x1)
#         x1 = self.cnn.layer1(x1)
#         x1 = self.cnn.layer2(x1)
#         x1 = self.cnn.layer3(x1)
#         x1 = self.cnn.layer4(x1)
#         x1 = self.cnn.avgpool(x1)

#         x1 = x1.squeeze()

#         return x1
        

# # Load models
# model_faune = MyModel()
# model_flore = MyModel()

# # Load models
# model_faune.cnn.load_state_dict(torch.load("./models/faune/model_resnet18_rgb_epoch_2.pth").state_dict())
# model_flore.cnn.load_state_dict(torch.load("./models/flore/model_resnet18_rgb_epoch_2.pth").state_dict())

# # Put the model in evaluation mode
# model_faune.eval()
# model_flore.eval()

# model_faune = model_faune.cuda()
# model_flore = model_flore.cuda()

# batch_size_test = 86
# total_test = len(df_features)//batch_size_test
# m = nn.Softmax(dim=1)
# stop = 0
# early_stop = 30

# data = ImageDataTest(df_env=df_features, data_path = DATA_PATH, load = load_patch, landcover_mapping=landcover_mapping)
# test_loader = DataLoader(dataset = data, shuffle=False, batch_size=batch_size_test)


# with tqdm(enumerate(test_loader), total=total_test) as tepoch_test:
#     for ii, (data, obs_id) in tepoch_test:
#         tepoch_test.set_description("Test prediction progress: ")
        
#         # Pass data to cuda and make a prediction
#         data = data.cuda()
#         output_faune = model_faune(data)            
#         output_flore = model_flore(data)           

#         # Convert the predictions to a numpy array
#         pred_faune = output_faune.cpu().detach().numpy()
#         pred_flore = output_flore.cpu().detach().numpy()

#         # Concatenate the two
#         pred = np.concatenate([pred_faune, pred_flore], axis=1)


#         # Add it to the ful test prediction
#         df_features.loc[obs_id,["cnn_"+str(i) for i in range(1024)]] = pred

#         # break
 
# df_features.to_csv("./enriched_df/df_features_coord_alt_land_resnet.csv")

# %%
n_estimators = 70
max_depth = 12

import os
from pathlib import Path

import pandas as pd
import numpy as np 

# Change this path to adapt to where you downloaded the data
DATA_PATH = Path("./geolifeclef-2022-lifeclef-2022-fgvc9")

# Create the path to save submission files
SUBMISSION_PATH = Path("./submissions")
os.makedirs(SUBMISSION_PATH, exist_ok=True)

# Clone the GitHub repository
# !rm -rf GLC
# !git clone https://github.com/maximiliense/GLC
    
    
# For evaluation and submission
from GLC.metrics import top_30_error_rate, top_k_error_rate_from_sets, predict_top_30_set
from GLC.submission import generate_submission_file

# For data loading and visualization
from GLC.data_loading.common import load_patch
from GLC.plotting import visualize_observation_patch
from GLC.data_loading.environmental_raster import PatchExtractor

print("***** Data loading started *****")
df_env = pd.read_csv("./enriched_df/df_features_coord_alt_land_resnet.csv", index_col="observation_id")


# We can finally compute the top 30 error rate on the val set
def predict_func(est, X):
    y_score = est.predict_proba(X)
    s_pred = predict_top_30_set(y_score)
    return s_pred            


# We define a batch predictor to take care of the memory
# as there are more than 17k classes
def batch_predict(predict_func, est, X_df, obs_id, batch_size=1024):
    res = predict_func(est, X_df.head(1).values)
    n_samples, n_outputs, dtype = X_df.shape[0], res.shape[1], res.dtype
    
    preds = np.empty((n_samples, n_outputs), dtype=dtype)
    print(preds.shape)
    
    for i in range(0, len(X_df), batch_size):
        obs_id_batch = obs_id[i:i+batch_size]
        X_batch = X_df.loc[obs_id_batch, :]
        
        # add_patch_info(X_batch, DATA_PATH=DATA_PATH, landcover_mapping=landcover_mapping)
        
        X_batch = X_batch.values
        
        preds[i:i+batch_size] = predict_func(est, X_batch)

        if (i/batch_size)%10 == 0:
            print("Prediction : " + str(100*(i+1)/len(X_df)) + "% completed")
            
    return preds


# Load train set of observations from France and USA and merge
df_obs_fr = pd.read_csv(DATA_PATH / "observations" / "observations_fr_train.csv", sep=";", index_col="observation_id")
df_obs_us = pd.read_csv(DATA_PATH / "observations" / "observations_us_train.csv", sep=";", index_col="observation_id")
df_obs = pd.concat((df_obs_fr, df_obs_us))

# Extract training and validation subsets as np arrays
obs_id_train = df_obs.index[df_obs["subset"] == "train"].values
obs_id_val = df_obs.index[df_obs["subset"] == "val"].values

# Separate values to predict
y_train = df_obs.loc[obs_id_train]["species_id"].values
y_val = df_obs.loc[obs_id_val]["species_id"].values

# Validation proportion
n_val = len(obs_id_val)
# print("Validation set size: {} ({:.1%} of train observations)".format(n_val, n_val / len(df_obs)))


# Same with test set of observations
df_obs_fr_test = pd.read_csv(DATA_PATH / "observations" / "observations_fr_test.csv", sep=";", index_col="observation_id")
df_obs_us_test = pd.read_csv(DATA_PATH / "observations" / "observations_us_test.csv", sep=";", index_col="observation_id")
df_obs_test = pd.concat((df_obs_fr_test, df_obs_us_test))

# Extract observaions as np array
obs_id_test = df_obs_test.index.values

# Define the train, val and test set as np arrays
X_train = df_env.loc[obs_id_train].values
X_val = df_env.loc[obs_id_val].values
X_test = df_env.loc[obs_id_test].values

y_train_df = df_obs.loc[obs_id_train]["species_id"]
X_train_df = df_env.loc[obs_id_train]

y_val_df = df_obs.loc[obs_id_val]["species_id"]
X_val_df = df_env.loc[obs_id_val]

X_test_df = df_env.loc[obs_id_test]


# Replace nan values with np.min
X_train_df.fillna(np.finfo(np.float32).min, inplace=True)
X_val_df.fillna(np.finfo(np.float32).min, inplace=True)
X_test_df.fillna(np.finfo(np.float32).min, inplace=True)


# Call a RF classifier, fit it on trainin set
# est = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, verbose=1)   
# params = {'max_depth':10, 'tree_method':'gpu_hist', }
params = {'tree_method':'gpu_hist', }
est = XGBClassifier(
                    # n_estimators=10,
                    # max_depth=4,
                    # learning_rate=0.05,
                    # subsample=0.9,
                    # colsample_bytree=0.9,
                    # missing=-999,
                    # random_state=42,
                    tree_method='gpu_hist',
                    verbose=3)

     
print("***** Fitting started *****")
est.fit(X_train, y_train)
print("***** Fitting successful *****\n")


# Validation
print("***** Batch predict started *****")
s_val = batch_predict(predict_func, est, X_val_df, obs_id_val)
print("***** Batch predict successful *****\n")

score_val = top_k_error_rate_from_sets(y_val, s_val)
print("Top-30 error rate: {:.1%}".format(score_val))


# Compute baseline on the test set
print("***** Batch predict test started *****")
s_pred = batch_predict(predict_func, est, X_test_df, obs_id_test)
print("***** Batch predict test successful *****\n")

# Generate the submission file
file = "./submissions/xgb_enriched_resnet_vect_" + str(n_estimators)+ "_est_" + str(max_depth) + "_max_dp"+ str(round(100*score_val)) +"_score.csv"
generate_submission_file(file, df_obs_test.index, s_pred)

