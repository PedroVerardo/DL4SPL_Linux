import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split

class ModelStatistics():
    def __init__(self,path_to_model_info: str, info_format: str, transpose: bool = False):
        self.format = info_format
        self.transpose = transpose
        self.path = path_to_model_info
        
        self.df = self.clean_and_reorder()
        self.activation_types = self.df['activation'].unique()
        self.loss_types = self.df['loss_function'].unique()
        self.dropout_types = self.df['dropout'].unique()
        self.architecture_types = self.df['architecture'].unique()
        self.optimizer_types = self.df['optimizer'].unique()
        
        
        
    def generate_dataframe(self, sep: str = ','):
        if self.format == 'pkl' or self.format == 'pickle':
            with open(path_to_models_statistics,"rb") as f:
                model_statistics = pickle.load(f)
            df = pd.DataFrame.from_dict(model_statistics)
            
        elif self.format == 'csv':
            df = pd.read_csv(path_to_model_info, sep = sep)
            
        elif self.format == 'hdf' or self.format == 'h5':
            df = pd.read_hdf(path_to_model_info)

        elif self.format == 'excel' or self.format == 'xls' or self.format == 'xlsx':
            try:
                df = pd.read_excel(path_to_model_info, sheet_name = 'Results')
            except Exception as e:
                print(e)
                print("Sheet name not found, trying to rename the \
                      first sheet to result, or check dependencies")
                return None
            
        elif self.format == 'parquet' or self.format == 'pq':
            df = pd.read_parquet(path_to_model_info)

            
        else:
            raise "Not suported format error"
            
        if self.transpose == True:
            df = df.T
        
        return df
    
    def standard_cleaner(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df_reset = df.reset_index()
        df_split = df_reset['index'].str.split('_', expand=True)
        
        df_split.drop(0, axis=1, inplace=True)
        df_split.drop(6, axis=1, inplace=True)
        
        df_split.rename(columns={1:'architecture', 2:'activation', 3:'dropout', 4:'optimizer', 5:'loss_function'}, inplace=True)
        
        df_split['dropout'] = df_split['dropout'].str.replace("Drop", "").astype(float)
        df_final = pd.concat([df_split, df_reset], axis = 1)
        
        df_final = df_final.astype({'architecture':str,
                                    'activation':str,
                                    'dropout':float,
                                    'optimizer':str,
                                    'loss_function':str,
                                    'index':str,
                                    'best_loss':float,
                                    'time':float,
                                    'total_epochs':int})
        return df_final
    
    
    def clean_and_reorder(self, personalized_cleaner_function = None) -> pd.DataFrame:
        
        df = self.generate_dataframe()
        
        if personalized_cleaner_function != None:
            df = personalized_cleaner_function(df)
        
        else:
            df = self.standard_cleaner(df)
        
        return df
    
        
    def get_top_paramters(self, best_loss_or_time:str, hyperparamiter_info: str, k: int):
        all_top_k = {}
        hyperparamiters = self.df[hyperparamiter_info].unique()
        
        for hyperparamiter in hyperparamiters:
            all_top_k_loss = {}
            for loss in self.loss_types:
                df = self.df[self.df['loss_function'] == loss]
                df = df[df[hyperparamiter_info] == hyperparamiter]
                lines = self.df.iloc[df[best_loss_or_time].nsmallest(k).index]
                all_top_k_loss[loss] = lines
            all_top_k[hyperparamiter] = all_top_k_loss
        
        return all_top_k
    
    def get_top_for_loss_type(self, best_loss_or_time:str):
        all_top_k_loss = {}
        for loss in self.loss_types:
            df = self.df[self.df['loss_function'] == loss]
            lines = self.df.iloc[df[best_loss_or_time].nsmallest(k).index]
            all_top_k_loss[loss] = lines
        return all_top_k_loss
    
    def get_cleaned_dataframe(self):
        return self.df
    
    
    def generate_dataframe_from_dict(self,info: dict):
        data = []
        for optimizer, metrics in info.items():
            for metric, details in metrics.items():
                data.append(details)
    
        return pd.concat(data, axis=0)
    
    def get_top_k_losses_for_optmizer(self, k:int = 5) -> pd.DataFrame:
        info = self.get_top_paramters('best_loss', 'optimizer', k)
    
        return self.generate_dataframe_from_dict(info)

    def get_not_learning_models_MAPE(self) -> pd.DataFrame:
        return self.df[(self.df["best_loss"] > 0.9) & (self.df["best_loss"] < 1.0)]
    
    def get_top_k_losses_for_dropout(self, k:int = 5) -> pd.DataFrame:
        info = self.get_top_paramters('best_loss', 'dropout', k)
        
        return self.generate_dataframe_from_dict(info)
    
    def get_top_k_losses_for_activation(self,  k:int = 5) -> pd.DataFrame:
        info = self.get_top_paramters('best_loss', 'activation', k)
    
        return self.generate_dataframe_from_dict(info)

    def get_top_k_losses_for_architecture(self,  k:int = 5) -> pd.DataFrame:
        info = self.get_top_paramters('best_loss', 'architecture', k)
        
        return self.generate_dataframe_from_dict(info)

    def get_top_k_losses_for_loss_function(self,  k:int = 5) -> pd.DataFrame:
        info = self.get_top_for_loss_type('best_loss', k)
        
        return self.generate_dataframe_from_dict(info)

    def get_top_k_times_for_optimizer(self,  k:int = 5) -> pd.DataFrame:
        info = self.get_top_paramters('time', 'optimizer', k)
        
        return self.generate_dataframe_from_dict(info)

    def get_top_k_times_for_dropout(self,  k:int = 5) -> pd.DataFrame:
        info = self.get_top_paramters('time', 'dropout', k)
        
        return self.generate_dataframe_from_dict(info)

    def get_top_k_times_for_activation(self,  k:int = 5) -> pd.DataFrame:
        info = self.get_top_paramters('time', 'activation', k)
        
        return self.generate_dataframe_from_dict(info)

    def get_top_k_times_for_architecture(self,  k:int = 5) -> pd.DataFrame:
        info = self.get_top_paramters('time', 'architecture', k)
        
        return self.generate_dataframe_from_dict(info)

    def get_top_k_times_for_loss_function(self,  k:int = 5) -> pd.DataFrame:
        info = self.get_top_for_loss_type('time', k)
        
        return self.generate_dataframe_from_dict(info)
    
    def get_avrg_time(self,list_group: list):
        df = self.df.groupby(list_group)['time'].mean().reset_index()
        return df.sort_values(by=['time'], ascending=True)