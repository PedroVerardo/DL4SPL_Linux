import copy
from torch import nn
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np

class LinuxDatasetObject(Dataset):
    def __init__(self, dataset, labels, device):
        self.dataset = dataset
        self.labels = labels

    def NumberOfFeatures(self):
        return self.dataset.shape[1]
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]

class EarlyStopping:
    def __init__(self, pation: int = 15, min_delta: int = 0):
        self.pation = pation
        self.min_delta = min_delta
        self.best_model = None
        self.best_loss = None
        self.counter = 0

    def __call__(self, model: nn.Module, val_loss: float, name_of_the_model_save: str = "result"):
        if self.best_loss == None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())

        elif val_loss <= self.best_loss:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0

        else:
            self.counter += 1
            if self.counter >= self.pation:
                model.load_state_dict(self.best_model)
                torch.save(self.best_model, f=name_of_the_model_save+'.pt')
                return True
            
        return False
    
    def reset(self):
        self.counter = 0
        self.best_model = None
        self.best_loss = None
    
    def save(self, model: nn.Module, name_of_the_model_save: str):
            model.load_state_dict(self.best_model)
            torch.save(self.best_model, f=name_of_the_model_save+'.pt')
    
def generate_train_test_samples( hdf_path: str, target_columns: list, columns_to_drop: list, device: str,
                                 test_size: float):
        try:
            df = pd.read_hdf(hdf_path)
        except:
            df = pd.read_csv(hdf_path)
            
        y = df[target_columns].to_numpy()
        X = df.drop(columns= columns_to_drop).to_numpy()
        
        X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train , X_val , y_train , y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

        X_train_tensor = torch.from_numpy(X_train).type(torch.float).to(device)
        X_test_tensor = torch.from_numpy(X_test).type(torch.float).to(device)
        X_validation_tensor = torch.from_numpy(X_val).type(torch.float).to(device)
        y_test_tensor = torch.from_numpy(y_test).type(torch.float).to(device).squeeze()
        y_train_tensor = torch.from_numpy(y_train).type(torch.float).to(device).squeeze()
        y_validation_tensor = torch.from_numpy(y_val).type(torch.float).to(device).squeeze()

        return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,X_validation_tensor,y_validation_tensor)

def load_data(device, featur_selectio_name: str):
    return generate_train_test_samples("data/"+featur_selectio_name+".h5", ["perf"],["perf", "active_options"], device, 0.2)

def train_dataloader(train_data: LinuxDatasetObject, batch_size: int = 4096, shuffle: bool = True):
    return DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)

def generate_data_loader_object(features, target, batch_size = None, device = 'cuda'):
    train = LinuxDatasetObject(features,target, device)
    if batch_size == None:
        return train_dataloader(train)
    return train_dataloader(train, batch_size)
