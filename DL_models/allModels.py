from torch import nn
import torch.nn.init as init
import numpy as np


'''
The main idea of this file is to create a class for each model that we want to test.
Each class will have the same structure, but with different number of layers and neurons in each layer.
Testing different architectures and see which one performs better.
'''

#-------------------------------Funnel--------------------------------------------------
'''
The Funnel model is a model that has a decreasing number of neurons in each layer.
Example:
                                   (ImpL)  
                                    /|\
                                   n n n
                                   \ | |
                                     n n
                                      \|
                                     (out)
ImpL -> Input Layer
n -> Neurons
out -> Output Layer
'''
class Model_Funnel1(nn.Module):
    def __init__(self, activation: nn, dropout_value: float, name: str):
        super().__init__()
        self.dropout_value = dropout_value
        self.name = name
        self.activation = activation

    def inicialize_values(self, features: int, classes: int):
        self.inputLayer = nn.Linear(features, int(features*2))
        self.activationLayer = self.activation
        self.dropout = nn.Dropout(self.dropout_value)
        self.hiddenLayer1 = nn.Linear(int(features*2),int((features*2)/12))
        self.outputLayer = nn.Linear(int((features*2)/12), classes)
        

    def get_name(self):
        return self.name

    def forward(self,x):
        z = self.inputLayer(x)
        z = self.activationLayer(self.hiddenLayer1(z))
        z = self.dropout(z)
        z = self.outputLayer(z)
        return z
    
class Model_Funnel2(nn.Module):
    def __init__(self, activation: nn, dropout_value: float, name: str):
        super().__init__()
        self.dropout_value = dropout_value
        self.name = name
        self.activation = activation

    def inicialize_values(self, features: int, classes: int):
        self.inputLayer = nn.Linear(features, int(features*2))
        self.activationLayer = self.activation
        self.dropout = nn.Dropout(self.dropout_value)
        self.hiddenLayer1 = nn.Linear(int(features*2),int((features*2)/6))
        self.hiddenLayer2 = nn.Linear(int(features*2/6),int((features*2)/12))
        self.outputLayer = nn.Linear(int((features*2)/12), classes)

    def get_name(self):
        return self.name

    def forward(self,x):
        z = self.inputLayer(x)
        z = self.hiddenLayer1(z)
        z = self.activationLayer(z)
        z = self.dropout(z)
        z = self.hiddenLayer2(z)
        z = self.activationLayer(z)
        z = self.dropout(z)
        z = self.outputLayer(z)
        return z
#-------------------------------Straight--------------------------------------------------
'''
The Straight model is a model that has the same number of neurons in each layer.
Example:
                                       (ImpL)  
                                        /|\
                                       n n n
                                       | | |
                                       n n n
                                       \ | /
                                       (out)
ImpL -> Input Layer
n -> Neurons
out -> Output Layer
'''
class Model_Straight1(nn.Module):
    def __init__(self, activation: nn, dropout_value: float, name: str):
        super().__init__()
        self.dropout_value = dropout_value
        self.name = name
        self.activation = activation
        
    def inicialize_values(self, features: int, classes: int):
        self.inputLayer = nn.Linear(features, int(features))
        self.activationLayer = self.activation
        self.dropout = nn.Dropout(self.dropout_value)
        self.hiddenLayer1 = nn.Linear(int(features),int((features)))
        self.outputLayer = nn.Linear(int((features)), classes)
        self.name = self.name

    def get_name(self):
        return self.name

    def forward(self,x):
        z = self.inputLayer(x)
        z = self.activationLayer(self.hiddenLayer1(z))
        z = self.dropout(z)
        z = self.outputLayer(z)
        return z
    
class Model_Straight2(nn.Module):
    def __init__(self, activation: nn, dropout_value: int, name: str):
        super().__init__()
        self.dropout_value = dropout_value
        self.name = name
        self.activation = activation
    
    def inicialize_values(self, features: int, classes: int):
        self.inputLayer = nn.Linear(features, int(features))
        self.activationLayer = self.activation
        self.hiddenLayer1 = nn.Linear(int(features),int((features)))
        self.hiddenLayer2 = nn.Linear(int(features),int((features)))
        self.dropout = nn.Dropout(self.dropout_value)
        self.outputLayer = nn.Linear(int((features)), classes)
        

    def get_name(self):
        return self.name

    def forward(self,x):
        z = self.inputLayer(x)
        z = self.activationLayer(self.hiddenLayer1(z))
        z = self.dropout(z)
        z = self.activationLayer(self.hiddenLayer2(z))
        z = self.dropout(z)
        z = self.outputLayer(z)
        return z
    
if __name__ == "__main__":
    pass
