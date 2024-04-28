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


###################
### DATA LOADER ###
###################
class ImageDataTest(Dataset):
    def __init__(self, df, df_env, data_path, load, landcover_mapping):
        super().__init__()
        self.df = df
        self.df_env = df_env
        self.data_path = data_path
        self.load = load
        self.landcover_mapping = landcover_mapping

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        obs_id = self.df.observation_id.iloc[idx]
        env_vect = self.df_env.loc[obs_id]
        patch = self.load(obs_id, self.data_path, landcover_mapping=self.landcover_mapping)
        patch = [patch[0].astype(np.float32)/255,
                 np.reshape(patch[1], (patch[1].shape[0], patch[1].shape[1],1)).astype(np.float32)/255,
                #   np.reshape(patch[2], (patch[2].shape[0], patch[2].shape[1],1)),
                   np.reshape(patch[3], (patch[3].shape[0], patch[3].shape[1],1)).astype(np.float32)/33]
        patch = torch.FloatTensor(np.concatenate(patch, axis=2))
        patch = torch.movedim(patch, 2, 0)
        return patch, torch.FloatTensor(env_vect)



def prediction_cnn():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Change this path to adapt to where you downloaded the data
    DATA_PATH = Path("./geolifeclef-2022-lifeclef-2022-fgvc9")

    # Same with test set of observations
    df_obs_fr_test = pd.read_csv(DATA_PATH / "observations" / "observations_fr_test.csv", sep=";", index_col="observation_id")
    df_obs_us_test = pd.read_csv(DATA_PATH / "observations" / "observations_us_test.csv", sep=";", index_col="observation_id")
    df_obs_test = pd.concat((df_obs_fr_test, df_obs_us_test))

    # Extract observaions as np array
    obs_id_test = df_obs_test.index.values
    df_test =  df_obs_test.loc[obs_id_test].reset_index(drop=False)

    # Load landcover metadata to use the patches
    df_suggested_landcover_alignment = pd.read_csv(DATA_PATH / "metadata" / "landcover_suggested_alignment.csv", sep=";")
    landcover_mapping = df_suggested_landcover_alignment["suggested_landcover_code"].values


    # Environmental features
    df_env = pd.read_csv("./enriched_df/df_features_coord_alt_land.csv", index_col="observation_id")
    df_env = (df_env-df_env.min())/(df_env.max()-df_env.min())   # min-max normalization


    SUBMISSION_PATH = "./submissions/"

    batch_size_test = 8
    total_test = len(df_test)//batch_size_test
    m = nn.Softmax(dim=1)
    stop = 0
    early_stop = 30

    test_data = ImageDataTest(df = df_test, df_env=df_env, data_path = DATA_PATH, load = load_patch, landcover_mapping=landcover_mapping, transform=normalize)
    test_loader = DataLoader(dataset = test_data, shuffle=False, batch_size=batch_size_test)
    preds = np.zeros([0,30])


    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.cnn = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
            
            self.cnn.conv1 = nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

            self.cnn.fc = nn.Sequential(
                                        nn.BatchNorm1d(num_features=512+31),
                                        nn.Linear(in_features=512 + 31 , out_features=4096, bias=True),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(in_features=4096, out_features=17037, bias=True)
                                                )

        def forward(self, patch, env_v):
            x1 = self.cnn.conv1(patch)
            x1 = self.cnn.bn1(x1)
            x1 = self.cnn.relu(x1)
            x1 = self.cnn.maxpool(x1)
            x1 = self.cnn.layer1(x1)
            x1 = self.cnn.layer2(x1)
            x1 = self.cnn.layer3(x1)
            x1 = self.cnn.layer4(x1)
            x1 = self.cnn.avgpool(x1)

            x1 = x1.squeeze()
            x2 = env_v

            x = torch.cat((x1, x2), dim=1)
            # x = self.cnn.fc(x)

            return x



    # Load models
    model_faune = MyModel()
    model_flore = MyModel()


    # Load models
    model_faune.cnn.load_state_dict(torch.load("./models/faune/model_resnet18_rgb_epoch_2.pth").state_dict())
    model_flore.cnn.load_state_dict(torch.load("./models/flore/model_resnet18_rgb_epoch_2.pth").state_dict())

    # Put the model in evaluation mode
    model_faune.eval()
    model_flore.eval()

    model_faune = model_faune.cuda()
    model_flore = model_flore.cuda()


    with tqdm(enumerate(test_loader), total=total_test) as tepoch_test:
        for ii, (data, env_v) in tepoch_test:
            tepoch_test.set_description("Test prediction progress: ")
            
            # Pass data to cuda and make a prediction
            data = data.cuda()
            env_v = env_v.cuda()

            output_faune = model_faune(data, env_v)                
            pred_faune = m(output_faune)

            output_flore = model_flore(data, env_v)                
            pred_flore = m(output_flore)

            # Convert the predictions to a numpy array
            pred_faune = pred_faune.cpu().detach().numpy()
            pred_flore = pred_flore.cpu().detach().numpy()

            # Keep the top 15 predictions for each kingdom
            pred_faune = predict_top_k_set(pred_faune, 15)
            pred_flore = predict_top_k_set(pred_flore, 15)

            # Concatenate the two
            pred = np.concatenate([pred_faune, pred_flore], axis=1)

            # Add it to the ful test prediction
            preds = np.concatenate([preds, pred], axis=0)