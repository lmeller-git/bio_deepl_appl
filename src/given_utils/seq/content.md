# Project: Protein stability prediction
  

  
In the project you will try to predict protein stability changes upon point mutations. 
  
We will use acuumulated data from experimental databases, i.e. the Megascale dataset. A current [pre-print paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10402116/) has already preprocessed the dataset and created homology reduced data splits. We will reuse these. To do so, download the data folder from [here](https://polybox.ethz.ch/index.php/s/txvcb5jKy1A0TbY) and unzip it.  
  

  
The data includes measurements of changes in the Gibbs free enrgy ($\Delta \Delta G $). 
  
This will be the value that you will have to predict for a given protein with a point mutation. 
  
As input data you can use the protein sequence or a protein embedding retreived from ESM, a state of the art protein model.  
  

  
Here we will use the sequence as input. 
  
The model will predict the $\Delta \Delta G $ of point mutations in this sequence. To make training more efficient, the model should directly predict the values for all possible mutations at each position in the sequence. So the expected output is a sequence of $ L \ (sequence \ length) \ x \ 20 \ (number \ amino \ acids)$. This will be shown in detail later.
  

  
Below we provide you with a strcuture for the project that you can start with.  
  
Edit the cells to your liking and add more code to create your final model.  
# Imports  
## Dataloading
  

  
We are using the Megascale dataset. The train, validation and test sets are already predefined.
  
We provide code to load the data and helper functions to encode the sequences to numerical vectors (one-hot encoding). You can use this code as a starting point, adjust it, or use your own data loading. 
  

  
Each sequence will be treated as one batch (easier to deal with different leghts this way). The class below returns a dictionary containing the one-hot encoded sequence of dimension $Lx20$, the target sequence of dimension $Lx20$, containing the $\Delta \Delta G $ values, and a mask of the same dimension which indicates with a 1 if an experimental value is available for that position. Only compute your loss on the positions where an experimental value is available. So compute your loss similar to this example:
  

  
``` 
  
prediction = model(x)
  
loss = loss(prediction[mask==1],labels[mask==1])
  
```  
## Model architecture and training
  

  
Now it's your turn. Create a model trained on the given sequences.  
  
Be aware that this is not a classification task, but a regression task. You want to predict continuous numbers as close to the measured $\Delta \Delta G $ value as possible.
  
You will need to adjust your architecture and loss accordingly.
  

  
Train the model with the predefined dataloaders and try to improve it. 
  
Only test on the test set at the very end, when you have finished fine-tuning you model.   
## Validation and visualization
  

  
To get a good feeling of how the model is performing and to compare with literature, compute the Pearson and Spearman correlations.
  
You can also plot the predictions in a scatterplot. We have added some code for that.   
