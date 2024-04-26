from torch import nn
import torch.nn.init as init
from allModels import *
import torch
import torchmetrics
import pickle
from time import time
from plots import *
from utils import *
import codecs
import tables
import os
import gc

class GridSearch():
    def __init__(self) -> None:
        self._activation_list: list = ["Sigmoid", "ELU", "ReLU", "ReLU6", "Tanh", "PReLU", "LeakyReLU", "RReLU"]
        self._architecture_list: list = ["Funnel1","Funnel2", "Straight1", "Straight2"]
        self._DropOut_list: list = ["Drop0.0","Drop0.1", "Drop0.15", "Drop0.2"]
        self._inicialization_list: list = ["uniform", "normal", "constant"]
        self._optimizer_list: list = ["Adam", "AdamW"]
        self._loss_list: list = ["MAPE", "MSE","SmoothL1Loss"]
        self._all_names: set = None
        self.saved_state = {}
        self.list_of_models: list[nn.Module] = []
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.epochs = 100
            
        (self.X_train_tensor,
         self.y_train_tensor,
         self.X_test_tensor, 
         self.y_test_tensor,
         self.X_validation_tensor,
         self.y_validation_tensor) = load_data(self.device, "dados")
        
        self.train_loader = generate_data_loader_object(self.X_train_tensor, self.y_train_tensor)

        self.es = EarlyStopping()
            
            
        directory = "savedBenchmarksInference"
        filename = "results.pkl"
        self.filepath = os.path.join(directory, filename)
        
        if not os.path.exists(directory):
            os.makedirs(directory)
            with open(self.filepath, 'wb') as f:
                pass
        else:
            with open(self.filepath, 'rb') as f:
                self.saved_state = pickle.load(f)
        
        
        directory_save_models = "savedModels"

        if not os.path.exists(directory_save_models):
            os.makedirs(directory_save_models)


    def _translate_activation_function(self, activation: str) -> nn:
        activation = activation.lower()
        if activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "elu":
            return nn.ELU()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "relu6":
            return nn.ReLU6()
        elif activation == "tanh":
             return nn.Tanh()
        elif activation == "prelu":
             return nn.PReLU()
        elif activation == "leakyrelu":
             return nn.LeakyReLU()
        elif activation == "rrelu":
             return nn.RReLU()
        else:
             return nn.ReLU()
        
    def _translate_dropout_value(self, dropout: str) -> float:
        dropout = dropout.lower()

        dropout_value = dropout[4:]

        if dropout_value == '':
            dropout_value = 0.0

        return float(dropout[4:])
    
    def _translate_optimizer(self,optomizer: str, model: nn.Module, learning_rate: float) -> torch.optim:
        optomizer = optomizer.lower()
        if optomizer == "adam":
            return torch.optim.Adam(model.parameters(), lr=learning_rate , fused=True)
        elif optomizer == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=learning_rate, fused=True)
        else:
            return torch.optim.Adam(model.parameters(), lr=learning_rate , fused=True)
    
    def _translate_loss(self, loss: str):
        loss = loss.lower()
        if loss == "mape":
            return torchmetrics.MeanAbsolutePercentageError()
        elif loss == "mse":
            return torchmetrics.MeanSquaredError()
        elif loss == "smoothl1loss":
            return nn.SmoothL1Loss()
        else:
            return torchmetrics.MeanAbsolutePercentageError()
            
    def _translate_model(self,architecture: str, activation: nn, dropout: nn, name: str):
        architecture = architecture.lower()
        if architecture == "funnel1":
            return Model_Funnel1(activation,dropout, name)
        elif architecture == "funnel2":
            return Model_Funnel2(activation,dropout, name)
        elif architecture == "straight1":
            return Model_Straight1(activation,dropout, name)
        elif architecture == "straight2":
            return Model_Straight2(activation,dropout, name)
        else:
            return Model_Funnel2(activation,dropout, name)

    def create_model_by_string(self, model_name: str) -> nn.Module:
        model_splited = model_name.split('_')
        architecture = model_splited[1]
        activation = model_splited[2]
        dropout = model_splited[3]

        activation = self._translate_activation_function(activation)
        dropout = self._translate_dropout_value(dropout)

        self.list_of_models.append(self._translate_model(architecture, activation, dropout, model_name))
    
    def generate_all_models(self):
        for name in self._all_names:
            self.create_model_by_string(name)

    def save_model_results(self, best_loss: float, name: str, time: float, total_epochs: int):


        
        self.saved_state[name] = {'best_loss': best_loss,
                                  'time':time,
                                  'total_epochs': total_epochs
                                  }
        
        

        with open(self.filepath, 'wb') as f:
            pickle.dump(self.saved_state, f)  

    def load_save_file(self, saved_file_path: str):
        with open(saved_file_path, 'rb') as f:
            self.saved_state = pickle.load(f)
        
        for key in self.saved_state.keys():
            if key in self._all_names:
                self._all_names.remove(key)

    def generate_random_model_strings(self):
        first = "Model_"
        all_names = set()
        for architecture in self._architecture_list:
            second = first + architecture + "_"
            for activ in self._activation_list:
                third = second + activ + "_"
                for dropout in self._DropOut_list:
                    fourth = third + dropout + "_"
                    for optim in self._optimizer_list:
                        fifth = fourth + optim + "_"
                        for loss in self._loss_list:
                            sixth = fifth + loss + "_"
                            all_names.add(sixth)
        self._all_names = all_names

    def get_total_combinations(self) -> int:
        """Function to get the number of total combinations of
        activation, architecture, Dropout and optimizers.

        Returns:
            int: Multiplication between the length of activation,
             architecture, Dropout and optimizers lists.
        """        
        return (len(self._activation_list) * 
                len(self._architecture_list) * 
                len(self._DropOut_list) * 
                len(self._optimizer_list) *
                len(self._loss_list))
    

    def train_model_with_inference(self, model: nn.Module):
        '''This function is just a regular PyTorch training pipeline;
        some constants are not being used to reduce server usage time.
        To display the results on the screen, simply uncomment what has been commented.
        '''
        
        loss_cont = []
        val_loss_cont = []

        model_name = model.name
        model_name_splited = model_name.split('_')

        model.inicialize_values(self.X_train_tensor.shape[1], 1)
        model.to(self.device)

        optimizer = self._translate_optimizer(model_name_splited[4], model, 0.001)
        loss_fn = self._translate_loss(model_name_splited[5]).to(self.device)

        star_time = time()
        
        save_path = "savedModels/" + model_name
        
        try:
            #print(model_name)
            for epoch in range(self.epochs):
                for batch_data, batch_labels in self.train_loader:
                    model.train()
                    y_pred = model(batch_data).squeeze()

                    loss = loss_fn(y_pred, batch_labels)

                    optimizer.zero_grad()

                    loss.backward()

                    optimizer.step()

                with torch.inference_mode():
                    #test_pred = model(self.X_test_tensor).squeeze()
                    #loss_test = loss_fn(test_pred, self.y_test_tensor)
                    val_pred = model(self.X_validation_tensor).squeeze()
                    loss_val = loss_fn(val_pred, self.y_validation_tensor)

                    loss_cont.append(torch.Tensor.cpu(loss).detach().numpy())
                    val_loss_in_cpu = torch.Tensor.cpu(loss_val).detach().numpy()
                    val_loss_cont.append(val_loss_in_cpu)

                    if self.es(model, val_loss_in_cpu, save_path):
                        self.save_model_results(self.es.best_loss, model_name,(time() - star_time), epoch)
                        break

#                 print(f"epoch {epoch}, train_loss = {loss}, test_loss = {loss_test}, val_loss = {loss_val}")
    #             print("allocate: " + str(torch.cuda.memory_allocated()/(10**9)) + " GB")
    #             print("cached: " + str(torch.cuda.memory_reserved()/(10**9)) + " GB")
            torch.save(model, "savedModels/" + model_name+".pt")
            self.save_model_results(self.es.best_loss, model_name,(time() - star_time), epoch)
            model.to('cpu')
            loss_fn.to('cpu')
            del loss_fn
            del optimizer
            del model
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            self.es.reset()
        except Exception as e:
            f = open("log.txt", "w")
            f.write(str(e))
            f.write(type(e))
            f.close()
           
        
    def get_actual_state(self):
        return self.saved_state

    def get_all_combination_results(self, inference: bool = True):
        for i, name in enumerate(self.all_names):
            if inference:
                self.train_model_with_inference(self.list_of_models[i])
            else:
                #To-Do
                '''To precisely analyze the training time of functions and measurements overall,
                it is necessary to isolate components to infer results from the model training phase.
                This involves excluding the time spent transferring data from GPU to CPU memory,
                resulting in a significant performance gain.
                '''
                pass

    @property
    def activation_list(self) -> list[str]:
        return self._activation_list
    
    @property
    def architecture_list(self) -> list[str]:
        return self._architecture_list
    
    @property
    def DropOut_list(self) -> list[str]:
        return self._DropOut_list
    
    @property
    def inicialization_list(self) -> list[str]:
        return self._inicialization_list
    
    @property
    def optimizer_list(self) -> list[str]:
        return self._optimizer_list
    
    @property
    def all_names(self) -> list[str]:
        return self._all_names
    
    @activation_list.setter
    def activation_list(self, str_list: list):
        self._activation_list = str_list
    
    @architecture_list.setter
    def architecture_list(self, str_list: list):
        self._architecture_list = str_list
    
    @DropOut_list.setter
    def DropOut_list(self, str_list: list):
        self._DropOut_list = str_list
    
    @inicialization_list.setter
    def inicialization_list(self, str_list: list):
        self._inicialization_list = str_list
    
    @optimizer_list.setter
    def optimizer_list(self, str_list: list):
        self._optimizer_list = str_list

    @all_names.setter
    def all_names(self, str_list: list):
        self._all_names = str_list

if __name__ == "__main__":
    grid = GridSearch()
    grid.generate_random_model_strings()
    grid.generate_all_models()
    grid.get_all_combination_results()
