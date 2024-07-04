# Summary

File Information
---
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

