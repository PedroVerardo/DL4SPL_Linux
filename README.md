# File Information
This section has a subsection for each file in the project, and each one has a brief explanation of the file and the importance of this file in the general context of the project. For more deeply understand, you can read the documentation of each file
### allModels
---
allModels.py contains all the neural network structures used on the project. This mainly represents the distribution of the neurons on each layer of the network.
This file is directed link to the file modelCreation.py.
Another interesting part is that you can notice these PyTorch classes have two different methods: "initialize_values" and "get_name."

+ inicialize_values -> The 'inicialize_values' method is designed to prevent memory issues by avoiding immediately loading the PyTorch model into memory. It's like drawing a draft of the model first, and then writing it later.
+ get_name -> we diferenciate each model by name, and with that name, we create the model by splitting the character "_" in this name

### modelCreation
---
The modelCreation.py file is the major file used in the project. Inside of that, you can notice the methods we use to create all names, separate all of these names in each hyperparameter we change in the paper, write the final model in GPU memory, and create the save logic of the time and the loss of each trained model.

### modelStatistics
---
The modelStatistics.py contains some of the major statistics we analyzed during the experiment. We tried to separate this module in an easy way that would allow us to accomplish it with other separated code we made in a notebook.

### modelPlots
---
The modelPlots.py uses the statistics gathered by the modelStatistics class to plot the information for the user. We designed it for running into notebooks.

# Notebooks
The notebooks are used for simple parts of the paper that do not require a lot of code development or use a lot of functions created in other files.

# How to run it
> Firstly, it is important to mention that the Model creation demands some weeks to run completely. I do not recommend running it if you do not have an RTX 3090 or a superior GPU from Nvidia. A simple approach you can take is to run only the best models in the singlePytorch.ipynb or singleTensorflow.ipynb to test only the best ones mentioned in the article.

Select singlePytorch or singleTensorflow notebooks to run the determinate model, changing the hyperparameters in that. Furthermore, download the Linux data gathered in the original paper[link].
<br>***OBS: to run the model creation function run the file main.py in the path DL_models/main.py***
