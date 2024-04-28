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
    def __init__(self, df, data_path, load, landcover_mapping, transform):
        super().__init__()
        self.df = df
        self.data_path = data_path
        self.load = load
        self.landcover_mapping = landcover_mapping
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        obs_id = self.df.observation_id.iloc[idx]
        species = self.df.species_id.iloc[idx]
        patch = self.load(obs_id, self.data_path, landcover_mapping=self.landcover_mapping)
        patch = [patch[0].astype(np.float32)/255,]
                #  np.reshape(patch[1], (patch[1].shape[0], patch[1].shape[1],1)).astype(np.float32)/255]
                #   np.reshape(patch[2], (patch[2].shape[0], patch[2].shape[1],1)).astype(np.float32)/4000,
                # np.reshape(patch[3], (patch[3].shape[0], patch[3].shape[1],1)).astype(np.float32)/33]
        patch = torch.FloatTensor(np.concatenate(patch, axis=2))
        patch = torch.movedim(patch, 2, 0)
        return self.transform(patch), torch.FloatTensor([species])


class ImageDataTest(Dataset):
    def __init__(self, df, data_path, load, landcover_mapping):
        super().__init__()
        self.df = df
        self.data_path = data_path
        self.load = load
        self.landcover_mapping = landcover_mapping
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        obs_id = self.df.observation_id.loc[idx]
        patch = self.load(obs_id, self.data_path, landcover_mapping=self.landcover_mapping)
        patch = [patch[0].astype(np.float32)/255,]
                #  np.reshape(patch[1], (patch[1].shape[0], patch[1].shape[1],1))/255],
                #   np.reshape(patch[2], (patch[2].shape[0], patch[2].shape[1],1)),
                #    np.reshape(patch[3], (patch[3].shape[0], patch[3].shape[1],1))]
        patch = torch.FloatTensor(np.concatenate(patch, axis=2))
        patch = torch.movedim(patch, 2, 0)
        return patch



#####################
### TRAINING LOOP ###
#####################

## %%time


torch.cuda.empty_cache()


def training():
    import torch 

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ### MODEl LDEFINITION ###
    # model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
    # model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)

    # print(model)
    # utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')


    model.eval().to(device)

    # Efficient net
    # model.stem = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True)
    
    
    # Resnet50
    #model.conv1 = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True)
    #torch.nn.init.xavier_normal(model.conv1.weight)

    model.classifier.fc = nn.Sequential(
        nn.Linear(in_features=1792, out_features=4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=17037)
    )

    # For resnet50
    # model.fc = nn.Sequential(
    #     nn.Linear(in_features=2048, out_features=4096),
    #     nn.ReLU(),
    #     #nn.Dropout(p=0.5),
    #     nn.Linear(in_features=4096, out_features=17037)
    # )

    #torch.nn.init.xavier_normal(model.fc[0].weight)

    # Unfreeze model weights
    for param in model.parameters():
        param.requires_grad = True
    
    # model = torch.load("./models/model_epoch_1_date_12_05.pth")

    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.01)  #they take 0.01
    loss_func = nn.CrossEntropyLoss()
    scheduler = lrs.StepLR(optimizer, step_size=3, gamma=0.1)


    # Change this path to adapt to where you downloaded the data
    DATA_PATH = Path("./geolifeclef-2022-lifeclef-2022-fgvc9")


    ### DATA LOADING ###
    # Load train set of observations from France and USA and merge
    df_obs_fr = pd.read_csv(DATA_PATH / "observations" / "observations_fr_train.csv", sep=";", index_col="observation_id")
    df_obs_us = pd.read_csv(DATA_PATH / "observations" / "observations_us_train.csv", sep=";", index_col="observation_id")
    df_obs = pd.concat((df_obs_fr, df_obs_us))

    # Extract training and validation subsets as np arrays
    obs_id_train = df_obs.index[df_obs["subset"] == "train"].values
    obs_id_val = df_obs.index[df_obs["subset"] == "val"].values

    df_train =  df_obs.loc[obs_id_train].reset_index(drop=False)
    df_val =  df_obs.loc[obs_id_val].reset_index(drop=False)

    # Load landcover metadata to use the patches
    df_suggested_landcover_alignment = pd.read_csv(DATA_PATH / "metadata" / "landcover_suggested_alignment.csv", sep=";")
    landcover_mapping = df_suggested_landcover_alignment["suggested_landcover_code"].values

    # User params
    epochs = 15
    batch_size = 16
    batch_size_val = 8
    num_workers=0
    plt_steps=100

    unique, _counts = np.unique(df_train.species_id, return_counts=True)
    
    # print(df_train.species_id.isin(unique[np.argsort(_counts)[::-1][:100]]).values)
    df_train = df_train[df_train.species_id.isin(unique[np.argsort(_counts)[::-1][:100]]).values]



    early_stop = 10

    # Normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    # Data loader call
    train_data = ImageData(df = df_train, data_path = DATA_PATH, load = load_patch, landcover_mapping=landcover_mapping, transform=normalize)
    train_loader = DataLoader(dataset = train_data, batch_size = batch_size, num_workers=num_workers, shuffle=True)

    val_data = ImageData(df = df_val, data_path = DATA_PATH, load = load_patch, landcover_mapping=landcover_mapping, transform=normalize)
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

    
    stop = 0

    total = len(df_train)//batch_size
    total_val = len(df_val)//batch_size_val
    m = nn.Softmax(dim=1)

    
    for epoch in range(epochs):    
        # Put the model in training mode (dropout layers unfreezed)
        model.train()    

        loss_log_train_cur = []
        acc_log_train_cur = []
        list_top_30_train_cur = []

        with tqdm(enumerate(train_loader), total=total) as tepoch:
            for ii, (data, target) in tepoch:
                tepoch.set_description(f"Epoch {epoch} training")

                target = target.squeeze().long()
                data, target = data.cuda(), target.cuda()
                
                optimizer.zero_grad()
                output = model(data)                

                # Compute loss
                loss = loss_func(output, target)
                loss.backward()

                # Update weights
                optimizer.step()  
                scheduler.step()

                if ii % plt_steps == 0:
                    loss_log_train_cur.append(loss.item())       

                if ii % (10*plt_steps) == 0:                    
                    # Show accuracy while training every x batches
                    pred = torch.argmax(m(output), dim=1)
                    acc = accuracy_score(target.cpu(), pred.cpu())
                    acc_log_train_cur.append(acc.item())       

                    top_30_error_train = top_30_error_rate(target.cpu().detach().numpy(), m(output).cpu().detach().numpy())
                    list_top_30_train_cur.append(top_30_error_train)


                tepoch.set_postfix(loss=loss.item(), accuracy=acc, top_30=top_30_error_train)

                # stop+=1
                # if stop > early_stop:
                #     stop = 0
                #     break

        # Save model before validation in case of memory overflow
        export_model = "./models/model_resnet_rgnir_epoch_"+str(epoch)+".pth"
        torch.save(model, export_model)


        # Put the model in evaluation mode
        model.eval()

        with tqdm(enumerate(val_loader), total=total_val) as tepoch_val:
            for ii, (data, target) in tepoch_val:
                tepoch_val.set_description(f"Epoch {epoch} validation")

                target = target.squeeze().long()
                data, target = data.cuda(), target.cuda()
                
                output = model(data)                

                # Compute validation loss and accuracy
                # if ii % plt_steps == 0:
                loss_val = loss_func(output, target)
                loss_log_val.append(loss_val.item())       
                
                # Show accuracy while training every x batches
                pred = torch.argmax(m(output), dim=1)
                acc_val = accuracy_score(target.cpu(), pred.cpu())
                acc_log_val.append(acc_val.item())       

                # top_30_val = predict_top_30_set(pred.cpu().detach().numpy())
                top_30_error_val = top_30_error_rate(target.cpu().detach().numpy(), m(output).cpu().detach().numpy())
                list_top_30_val.append(top_30_error_val)


                tepoch_val.set_postfix(val_loss=loss_val.item(), val_accuracy=acc_val, val_top_30=top_30_error_val)

                # stop+=1
                # if stop > early_stop:
                #     stop = 0
                #     break

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
        export_loss_png = "./models/loss_resnet_rgnir_epochs_"+str(epochs)+"_batch_size_"+str(batch_size)+".png"
        x_train = range(len(loss_log_train))
        x_train_acc = range(len(acc_log_train))
        x = range(epoch+1)
        
        # fig = plt.figure(figsize=(14,8))    
        fig, axs = plt.subplots(2,3, figsize=(17,10), constrained_layout=True)
        # plt.subplot(231)
        axs[0,0].plot(x_train, loss_log_train, label="train_loss per batch")
        axs[0,0].legend()
        
        # plt.subplot(234)
        axs[1,0].plot(x, loss_log_train_epoch, label="train_loss per epoch", marker="o")
        axs[1,0].set_ylabel('train_loss')
        axs[1,0].legend(loc="upper left")
        
        ax_b = axs[1,0].twinx()
        ax_b.plot(x, loss_log_val_epoch, label="val_loss per epoch", marker="o", color='orange')
        ax_b.set_ylabel('val_loss')
        ax_b.legend(loc="upper right")
        
        # plt.subplot(232)
        axs[0,1].plot(x_train_acc, acc_log_train, label="train_acc per batch")
        axs[0,1].legend()
        
        # plt.subplot(235)
        axs[1,1].plot(x, acc_log_train_epoch, label="train_acc per epoch", marker="o")
        axs[1,1].plot(x, acc_log_val_epoch, label="val_acc per epoch", marker="o")
        axs[1,1].legend()
        
        
        # plt.subplot(233)
        axs[0,2].plot(x_train_acc, list_top_30_train, label="train_top_30_error per batch")
        axs[0,2].legend()
        
        # plt.subplot(236)
        axs[1,2].plot(x, list_top_30_train_epoch, label="train_top_30_error per epoch", marker="o")
        axs[1,2].plot(x, list_top_30_val_epoch, label="val_top_30_error per epoch", marker="o")
        axs[1,2].legend()
        
        
        plt.suptitle("Training metrics")
        
        fig.patch.set_facecolor('white')
        fig.savefig(export_loss_png, transparent=False)




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


    SUBMISSION_PATH = "./submissions/"

    batch_size_test = 32
    total_test = len(df_test)//batch_size_test
    m = nn.Softmax(dim=1)
    stop = 0
    early_stop = 30

    test_data = ImageDataTest(df = df_test, data_path = DATA_PATH, load = load_patch, landcover_mapping=landcover_mapping)
    test_loader = DataLoader(dataset = test_data, shuffle=False, batch_size=batch_size_test)
    preds = np.zeros([0,30])

    # Load model
    model = torch.load("./models/model_epoch_1_date_12_05.pth")

    # Put the model in evaluation mode
    model.eval()


    with tqdm(enumerate(test_loader), total=total_test) as tepoch_test:
        for ii, data in tepoch_test:
            tepoch_test.set_description("Test prediction progress: ")
            
            # Pass data to cuda and make a prediction
            data = data.cuda()
            output = model(data)                
            pred = m(output)

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