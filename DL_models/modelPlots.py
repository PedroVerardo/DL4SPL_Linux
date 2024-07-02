import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np

from modelStatistics import ModelStatistics

class ModelPlots(ModelStatistics):
    def __init__(self, path_to_model_info: str, path_to_model: str, path_to_original_data: str, info_format: str, transpose: bool = False):
        super().__init__(path_to_model_info, info_format, transpose)
        self.path_to_model = path_to_model
        self.original_df = self.generate_dataframe(path_to_original_data)
        self.y = self.original_df["perf"]
        self.original_df.drop(columns= ["perf", "active_options"], inplace = True)
        self.x = self.original_df
        self.X_train , self.X_test , self.y_train , self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
        self.X_train , self.X_val , self.y_train , self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.1, random_state=42)
        
        
    def plot_top_k_losses_for_optmizer(self, k:int = 5):
        return self.get_top_k_losses_for_optmizer(k)
        
    def plot_model_prediction_in_actual_data(self, model_name: str):
        plt.plot(self.y_test[:100], color = 'b')
        model = torch.load(self.path_to_model+model_name)
        tensor = torch.from_numpy(self.X_test[:100].to_numpy()).type(torch.float).to('cuda')
        pred = model(tensor)
        pred = torch.Tensor.cpu(pred).detach().numpy()
        plt.plot(pred, color='r', linestyle = 'dashed')
        plt.show()
    
    def plot_results():
        plt.figure(figsize=(20,6))
        plt.subplot(1,3,1)
        ax = sns.lineplot(y=np.array(train_loss), x=np.array(epoch), label="loss", palette="binary")
        ax.set(xlabel= 'epochs', ylabel= 'train_loss')
        plt.title("Train Loss")
        #----------------------------------------------------------------
        plt.subplot(1,3,2)
        ax = sns.lineplot(y=np.array(train_loss), x=np.array(epoch), label="test loss", palette="flare")
        ax.set(xlabel= 'epochs', ylabel= 'train_loss')
        ax = sns.lineplot(x=np.array(epoch), y=np.array(validation_loss), label="Loss Convergence", color="red")
        ax.set(xlabel= 'epochs', ylabel= 'validation_loss')
        plt.title("Train Vs Validation")
        #----------------------------------------------------------------
        plt.subplot(1,3,3)
        absolut_loss = np.subtract(validation_loss, train_loss)
        ax = sns.lineplot(x=np.array(epoch), y=absolut_loss, label="Loss Convergence", color="red")
        ax.set(xlabel= 'epochs', ylabel= 'Vloss - Tloss')
        plt.title("Loss Convergence")
        
    def print_df(self):
        return self.original_df