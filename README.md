# File Information
This section includes a subsection for each file in the project, providing a brief explanation of the file and its significance within the overall context of the project. For a deeper understanding, you can refer to the documentation for each file.
### allModels.py
---
The `allModels.py` file contains all the neural network architectures used in the project. It primarily defines the distribution of neurons in each layer of the network.
This file is directed link to the `modelCreation.py` file.
Another interesting part is that you can notice these PyTorch classes have two different methods: *initialize_values* and *get_name*.

+ `inicialize_values`: This method is designed to prevent memory issues by delaying the loading of the PyTorch model into memory. It's similar to outlining a draft of the model first, and then fully implementing it later.
+ `get_name`: We diferenciate each model by name, and with that name, we create the model by splitting the character "_" in this name

### modelCreation.py
---
The `modelCreation.py` file is a key component of the project. It contains methods for generating all model names, separating these names based on each hyperparameter modified in the paper, writing the final model to GPU memory, and implementing the logic for saving the time and loss of each trained model.

### modelStatistics.py
---
The `modelStatistics.py` file contains key statistics analyzed during the experiment. We designed this module to be easily integrated with other code segments created in a notebook, facilitating streamlined analysis and experimentation.

### modelPlots.py
---
The `modelPlots.py` file utilizes the statistics gathered by the `modelStatistics` class to create visualizations for the user. It is specifically designed to be executed within notebooks.

# Notebooks
The notebooks are used for simple parts of the paper that do not require a lot of code development or use a lot of functions created in other files.

## How to run it
> Firstly, it is important to note that the model creation process can take several weeks to complete. We do not recommend running it unless you have an NVIDIA RTX 3090 or a superior GPU. A simpler approach is to run only the best models using `singlePytorch.ipynb` or `singleTensorflow.ipynb` to test the top-performing models mentioned in the article.
- Open a new terminal to install the dependencies of the project using `pip install -r requirements.txt`
- Select `singlePytorch` or `singleTensorflow` notebooks to run the determinate model, changing the hyperparameters in that. Furthermore, download the Linux [dataset](https://zenodo.org/records/4943884#.YqG5cTlByV4) gathered in the original [paper](https://dl.acm.org/doi/abs/10.1145/3546932.3546997?casa_token=5GnRV_DUr_wAAAAA:MxyEVvluTm-3ExRjvfnh64LJlWI1e6ii9_Ht5n9eNGDdGaINTwYctEjo58IwVNYUPwn_rS0LwpMK).

<br>***OBS: to run the model creation function run the file main.py in the path DL_models/main.py***
