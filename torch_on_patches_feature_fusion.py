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

# For evaluation and submission
from GLC.metrics import top_30_error_rate, predict_top_30_set #,top_k_error_rate_from_sets
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
class ImageData(Dataset):
    def __init__(self, df, df_env, data_path, load, landcover_mapping, transform):
        super().__init__()
        self.df = df
        self.df_env = df_env
        self.data_path = data_path
        self.load = load
        self.landcover_mapping = landcover_mapping
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        obs_id = self.df.observation_id.iloc[idx]
        species = self.df.species_id.iloc[idx]
        env_vect = self.df_env.loc[obs_id]
        patch = self.load(obs_id, self.data_path, landcover_mapping=self.landcover_mapping)
        patch = [patch[0].astype(np.float32)/255,
                 np.reshape(patch[1], (patch[1].shape[0], patch[1].shape[1],1)).astype(np.float32)/255]
                #   np.reshape(patch[2], (patch[2].shape[0], patch[2].shape[1],1)).astype(np.float32)/4000,
                # np.reshape(patch[3], (patch[3].shape[0], patch[3].shape[1],1)).astype(np.float32)/33]
        patch = torch.FloatTensor(np.concatenate(patch, axis=2))
        patch = torch.movedim(patch, 2, 0)
        return self.transform(patch), torch.FloatTensor(env_vect), torch.FloatTensor([species])


class ImageDataTest(Dataset):
    def __init__(self, df, df_env, data_path, load, landcover_mapping, transform):
        super().__init__()
        self.df = df
        self.df_env = df_env
        self.data_path = data_path
        self.load = load
        self.landcover_mapping = landcover_mapping
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        obs_id = self.df.observation_id.iloc[idx]
        env_vect = self.df_env.loc[obs_id]
        patch = self.load(obs_id, self.data_path, landcover_mapping=self.landcover_mapping)
        patch = [patch[0].astype(np.float32)/255,
                 np.reshape(patch[1], (patch[1].shape[0], patch[1].shape[1],1)).astype(np.float32)/255]
                #   np.reshape(patch[2], (patch[2].shape[0], patch[2].shape[1],1)),
                #    np.reshape(patch[3], (patch[3].shape[0], patch[3].shape[1],1))]
        patch = torch.FloatTensor(np.concatenate(patch, axis=2))
        patch = torch.movedim(patch, 2, 0)
        return self.transform(patch), torch.FloatTensor(env_vect)



#####################
### TRAINING LOOP ###
#####################

## %%time


torch.cuda.empty_cache()


def training():
    import torch 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Change this path to adapt to where you downloaded the data
    DATA_PATH = Path("./geolifeclef-2022-lifeclef-2022-fgvc9")


    # User params
    epochs = 1
    lr_start = 0.001
    batch_size = 16
    batch_size_val = 8
    num_workers=0
    plt_steps=30

    # Early stopping parameters
    early_stop = 100000000
    early_stop_th = 0.76
    patience = 5

    # Data params
    nb_classes = 10000

    ### DATA LOADING ###
    # Load landcover metadata to use the patches
    df_suggested_landcover_alignment = pd.read_csv(DATA_PATH / "metadata" / "landcover_suggested_alignment.csv", sep=";")
    landcover_mapping = df_suggested_landcover_alignment["suggested_landcover_code"].values

    # Environmental features
    df_env = pd.read_csv("./enriched_df/df_features_coord_alt_land.csv", index_col="observation_id")
    df_env = (df_env-df_env.min())/(df_env.max()-df_env.min())   # min-max normalization
    # df_env = (df_env-df_env.mean())/df_env.std() # mean-std normalization

    
    # Species information
    df_species = pd.read_csv("./geolifeclef-2022-lifeclef-2022-fgvc9/metadata/species_details.csv", sep=";")#, index_col="species_id")
    species = df_species.GBIF_kingdom_name.values


    # Load train set of observations from France and USA and merge
    df_obs_fr = pd.read_csv(DATA_PATH / "observations" / "observations_fr_train.csv", sep=";", index_col="observation_id")
    df_obs_us = pd.read_csv(DATA_PATH / "observations" / "observations_us_train.csv", sep=";", index_col="observation_id")
    df_obs = pd.concat((df_obs_fr, df_obs_us))

    # Extract species kingdom
    # df_obs['kingdom'] = df_obs.apply(lambda x: df_species.loc[x.species_id].GBIF_kingdom_name, axis=1)
    df_obs['kingdom'] = df_obs.apply(lambda x: species[x.species_id], axis=1)
 

    # Extract training and validation subsets as np arrays
    obs_id_train = df_obs.index[df_obs["subset"] == "train"].values
    obs_id_val = df_obs.index[df_obs["subset"] == "val"].values

    df_train =  df_obs.loc[obs_id_train].reset_index(drop=False)
    df_val =  df_obs.loc[obs_id_val].reset_index(drop=False)

    df_train_faune = df_train[df_train['kingdom']=='Animalia']
    df_train_flore = df_train[df_train['kingdom']=='Plantae']
    df_val_faune = df_val[df_val['kingdom']=='Animalia']
    df_val_flore = df_val[df_val['kingdom']=='Plantae']

    
    # Keep most populated classes
    # unique, _counts = np.unique(df_train.species_id, return_counts=True)
    # df_train = df_train[df_train.species_id.isin(unique[np.argsort(_counts)[::-1][:100]]).values]
    unique, _counts = np.unique(df_train_faune.species_id, return_counts=True)
    df_train_faune = df_train_faune[df_train_faune.species_id.isin(unique[np.argsort(_counts)[::-1][:nb_classes]]).values]
    
    unique, _counts = np.unique(df_train_flore.species_id, return_counts=True)
    df_train_flore = df_train_flore[df_train_flore.species_id.isin(unique[np.argsort(_counts)[::-1][:nb_classes]]).values]


    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.cnn = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)
            
            self.cnn.stem = nn.Conv2d(4, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True)

            self.cnn.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=1),
                                                nn.Flatten(),
                                                nn.Dropout(p=0.4, inplace=False),
                                                )

            self.cnn.fc = nn.Sequential(
                                        nn.BatchNorm1d(num_features=1792+31),
                                        nn.Linear(in_features=1792 + 31 , out_features=4096),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(in_features=4096, out_features=17037)
                                                )

        def forward(self, patch, env_v):
            x1 = self.cnn.stem(patch)
            x1 = self.cnn.layers(x1)
            x1 = self.cnn.features(x1)
            x1 = self.cnn.classifier(x1)

            x2 = env_v

            x = torch.cat((x1, x2), dim=1)
            x = self.cnn.fc(x)

            return x


    def kingdom_training(df_train, df_val, df_env, export_folder="./models/", es=10000, es_th=early_stop_th, lr=0.01):

        ### MODEl LDEFINITION ###
        model = MyModel()

        model.eval().to(device)

        # Unfreeze model weights
        for param in model.parameters():
            param.requires_grad = True
        
        # model = torch.load("./models/model_epoch_1_date_12_05.pth")

        model = model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=lr)  #they take 0.01
        loss_func = nn.CrossEntropyLoss()
        scheduler = lrs.StepLR(optimizer, step_size=1, gamma=1)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.450],
                                        std=[0.229, 0.224, 0.225, 0.225])


        # Data loader call
        train_data = ImageData(df = df_train, df_env=df_env, data_path = DATA_PATH, load = load_patch, landcover_mapping=landcover_mapping, transform=normalize)
        train_loader = DataLoader(dataset = train_data, batch_size = batch_size, num_workers=num_workers, shuffle=True)

        val_data = ImageData(df = df_val, df_env=df_env, data_path = DATA_PATH, load = load_patch, landcover_mapping=landcover_mapping, transform=normalize)
        val_loader = DataLoader(dataset = val_data, batch_size=batch_size_val, shuffle=True)


        # Train model
        loss_log_train = []
        acc_log_train = []
        list_top_30_train = []

        loss_log_val = []
        acc_log_val = []
        list_top_30_val = []
        
        loss_log_train_epoch = []
        acc_log_train_epoch = []
        list_top_30_train_epoch = []
        
        loss_log_val_epoch = []
        acc_log_val_epoch = []
        list_top_30_val_epoch = []

    
        total = len(df_train)//batch_size
        total_val = len(df_val)//batch_size_val
        m = nn.Softmax(dim=1)

        for epoch in range(epochs):    
            stop = 0
            trigger_times = 0
    
        # Put the model in training mode (dropout layers unfreezed)
            model.train()    

            loss_log_train_cur = []
            acc_log_train_cur = []
            list_top_30_train_cur = []

            with tqdm(enumerate(train_loader), total=total) as tepoch:
                for ii, (data, env_v, target) in tepoch:
                    tepoch.set_description(f"Epoch {epoch} training")

                    target = target.squeeze().long()
                    data, env_v, target = data.cuda(), env_v.cuda(), target.cuda()

                    optimizer.zero_grad()
                    output = model(data,env_v)                

                    # Compute loss
                    loss = loss_func(output, target)
                    loss.backward()

                    # Update weights
                    optimizer.step()  
                    scheduler.step()

                    if ii % plt_steps == 0:
                        loss_log_train_cur.append(loss.item())       

                    # if ii % (10*plt_steps) == 0:                    
                        # Show accuracy while training every x batches
                        pred = torch.argmax(m(output), dim=1)
                        acc = accuracy_score(target.cpu(), pred.cpu())
                        acc_log_train_cur.append(acc.item())       

                        top_30_error_train = top_30_error_rate(target.cpu().detach().numpy(), m(output).cpu().detach().numpy())
                        list_top_30_train_cur.append(top_30_error_train)


                        ### Early stopping ###
                        # Based on top-30 error rate
                        if top_30_error_train < es_th :
                            trigger_times += 1
                            if trigger_times >= patience:
                                # print('\nEarly stopping!\nStart validation.')
                                break

                        else:   
                            trigger_times = 0
                

                    tepoch.set_postfix(loss=loss.item(), accuracy=acc, top_30=top_30_error_train, trigs=trigger_times)

                    # For short training (debug)
                    stop+=1
                    if stop > es:
                        stop = 0
                        break


            # Save model before validation in case of memory overflow
            export_model = export_folder + "model_effnet_rgnir_epoch_"+str(epoch)+".pth"
            torch.save(model.cnn, export_model)


            # Put the model in evaluation mode
            model.eval()

            with tqdm(enumerate(val_loader), total=total_val) as tepoch_val:
                for ii, (data, env_v, target) in tepoch_val:
                    tepoch_val.set_description(f"Epoch {epoch} validation")

                    target = target.squeeze().long()
                    data, env_v, target = data.cuda(), env_v.cuda(), target.cuda()
                    
                    output = model(data, env_v)                

                    try:
                        # Compute validation loss and accuracy
                        # if ii % plt_steps == 0:
                        loss_val = loss_func(output, target)
                        loss_log_val.append(loss_val.item())       
                        
                        # Show accuracy while training every x batches
                        pred = torch.argmax(m(output), dim=1)
                        acc_val = accuracy_score(target.cpu(), pred.cpu())
                        acc_log_val.append(acc_val.item())       

                        top_30_error_val = top_30_error_rate(target.cpu().detach().numpy(), m(output).cpu().detach().numpy())
                        list_top_30_val.append(top_30_error_val)
                    except:
                        pass

                    tepoch_val.set_postfix(val_loss=loss_val.item(), val_accuracy=acc_val, val_top_30=top_30_error_val)

                    stop+=1
                    if stop > es:
                        stop = 0
                        break


            loss_log_train += loss_log_train_cur
            acc_log_train += acc_log_train_cur
            list_top_30_train += list_top_30_train_cur
            
            loss_log_train_epoch.append(np.mean(loss_log_train_cur))
            acc_log_train_epoch.append(np.mean(acc_log_train_cur))
            list_top_30_train_epoch.append(np.mean(list_top_30_train_cur))

            loss_log_val_epoch.append(np.mean(loss_log_val))
            acc_log_val_epoch.append(np.mean(acc_log_val))
            list_top_30_val_epoch.append(np.mean(list_top_30_val))


            print('Epoch: {} - Loss (train): {:.6f}'.format(epoch, np.mean(loss_log_train_cur)),
                    " - Accuracy (train): {:.6f}".format(np.mean(acc_log_train_cur)),
                    " - Top 30 error (train): {:.6f}".format(np.mean(list_top_30_train_cur)),
                    ' - Loss (val): {:.6f}'.format(np.mean(loss_log_val)),
                    " - Accuracy (val): {:.6f}".format(np.mean(acc_log_val)),
                    " - Top 30 error (val): {:.6f}".format(np.mean(list_top_30_val)))


            # Visualization monitoring
            export_loss_png = export_folder+"loss_effnet_rgnir_epochs_"+str(epochs)+"_batch_size_"+str(batch_size)+".png"
            x_train = range(len(loss_log_train))
            x_train_acc = range(len(acc_log_train))
            x = range(epoch+1)
            
            fig, axs = plt.subplots(2,3, figsize=(17,10), constrained_layout=True)
            axs[0,0].plot(x_train, loss_log_train, label="train_loss per batch")
            axs[0,0].legend()
            
            axs[1,0].plot(x, loss_log_train_epoch, label="train_loss per epoch", marker="o")
            axs[1,0].set_ylabel('train_loss')
            axs[1,0].legend(loc="upper left")
            
            ax_b = axs[1,0].twinx()
            ax_b.plot(x, loss_log_val_epoch, label="val_loss per epoch", marker="o", color='orange')
            ax_b.set_ylabel('val_loss')
            ax_b.legend(loc="upper right")
            
            axs[0,1].plot(x_train_acc, acc_log_train, label="train_acc per batch")
            axs[0,1].legend()
            
            axs[1,1].plot(x, acc_log_train_epoch, label="train_acc per epoch", marker="o")
            axs[1,1].plot(x, acc_log_val_epoch, label="val_acc per epoch", marker="o")
            axs[1,1].legend()
            
            
            axs[0,2].plot(x_train_acc, list_top_30_train, label="train_top_30_error per batch")
            axs[0,2].legend()
            
            axs[1,2].plot(x, list_top_30_train_epoch, label="train_top_30_error per epoch", marker="o")
            axs[1,2].plot(x, list_top_30_val_epoch, label="val_top_30_error per epoch", marker="o")
            axs[1,2].legend()
            
            
            plt.suptitle("Training metrics")
            
            fig.patch.set_facecolor('white')
            fig.savefig(export_loss_png, transparent=False)


    print("|||| FAUNA TRAINING ||||")
    kingdom_training(df_train_faune, df_val_faune, df_env, export_folder="./models/faune/", es=10000, lr=0.01)

    print("|||| FLORA TRAINING ||||")
    kingdom_training(df_train_flore, df_val_flore, df_env, export_folder="./models/flore/",es=20000, lr=0.001)



#######################
### TEST PREDICTION ###
#######################

def prediction():

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
    # df_landcover_labels = pd.read_csv(DATA_PATH / "metadata" / "landcover_original_labels.csv", sep=";")
    df_suggested_landcover_alignment = pd.read_csv(DATA_PATH / "metadata" / "landcover_suggested_alignment.csv", sep=";")
    landcover_mapping = df_suggested_landcover_alignment["suggested_landcover_code"].values


    # Environmental features
    df_env = pd.read_csv("./enriched_df/df_features_coord_alt_land.csv", index_col="observation_id")
    df_env = (df_env-df_env.min())/(df_env.max()-df_env.min())   # min-max normalization
    # df_env = (df_env-df_env.mean())/df_env.std() # mean-std normalization


    SUBMISSION_PATH = "./submissions/"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.450],
                                        std=[0.229, 0.224, 0.225, 0.225])


    batch_size_test = 32
    total_test = len(df_test)//batch_size_test
    m = nn.Softmax(dim=1)
    stop = 0
    early_stop = 30

    test_data = ImageDataTest(df = df_test, df_env=df_env, data_path = DATA_PATH, load = load_patch, landcover_mapping=landcover_mapping, transform=normalize)
    test_loader = DataLoader(dataset = test_data, shuffle=False, batch_size=batch_size_test)
    preds = np.zeros([0,30])

    # Load models
    model_faune = torch.load("./models/faune/model_epoch_1_date_12_05.pth")
    model_flore = torch.load("./models/flore/model_epoch_1_date_12_05.pth")

    # Put the model in evaluation mode
    model_faune.eval()
    model_flore.eval()


    with tqdm(enumerate(test_loader), total=total_test) as tepoch_test:
        for ii, (data, env_v) in tepoch_test:
            tepoch_test.set_description("Test prediction progress: ")
            
            # Pass data to cuda and make a prediction
            data = data.cuda()
            env_v = env_v.cuda()

            output_faune = model_faune(data, env_v)                
            pred = m(output_faune)

            output_flore = model_flore(data, env_v)                
            pred_flore = m(output_flore)


            # Convert the prediction to a numpy array
            pred = pred.cpu().detach().numpy()
            pred = predict_top_30_set(pred)
            preds = np.concatenate([preds, pred], axis=0)

            # #Early stopping
            # if stop > early_stop:
            #     stop = 0
            #     break
            # else:
            #     stop+=1

        preds = np.array(preds).astype(np.int32)



    # Generate the submission file
    generate_submission_file(SUBMISSION_PATH + "efficient_net_1_epochs.csv", df_obs_test.index[:preds.shape[0]], preds)



##################
### RUN SCRIPT ###
##################
if __name__ == '__main__':
    training()