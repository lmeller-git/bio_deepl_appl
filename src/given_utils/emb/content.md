# Project: Protein stability prediction
  

  
In the project you will try to predict protein stability changes upon point mutations. 
  
We will use acuumulated data from experimental databases, i.e. the Megascale dataset. A current [pre-print paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10402116/) has already preprocessed the dataset and created homology reduced data splits. We will reuse these. To do so, download the data folder from [here](https://polybox.ethz.ch/index.php/s/txvcb5jKy1A0TbY) and unzip it.  
  

  
The data includes measurements of changes in the Gibbs free enrgy ($\Delta \Delta G $). 
  
This will be the value that you will have to predict for a given protein with a point mutation. 
  
As input data you can use the protein sequence or a protein embedding retreived from ESM, a state of the art protein model.  
  

  
Here we will use protein embeddings computed by ESM as input. 
  
We provide precomputed embeddings from the last layer of the smallest ESM model. You can adjust the Dataloader's code to load the embedding of the wild type or of the mutaed sequence or both. You can use it however you like. This is just to provide you easy access to embeddings. If you want to compute your own embeddings from other layers or models you can do that, too. 
  

  
Below we provide you with a strcuture for the project that you can start with.  
  
Edit the cells to your liking and add more code to create your final model.  
## Imports  
## Dataloading
  

  
We are using the Megascale dataset. The train, validation and test sets are already predefined.  
  
As mentioned, we provide embeddings from the last layer of ESM as input. You can access either the wild type or the mutated sequence and you could also further adjsut the embeddings. 
  
Here we have an embedding representing the complete sequence. It was computed by averaging over the embeddings per residue in the sequence. 
  

  
The ``Dataset`` classes return tuples of ``(embedding, ddg_value)``.  
## Model architecture and training
  

  
Now it's your turn. Create a model trained on the embeddings and the corresponding ddG values.  
  
Be aware that this is not a classification task, but a regression task. You want to predict a continuous number that is as close to the measured $\Delta \Delta G $ value as possible.
  
You will need to adjust your architecture and loss accordingly.
  

  
Train the model with the predefined dataloaders. And try to improve the model. 
  
Only test on the test set at the very end, when you have finished fine-tuning you model.   
## Validation and visualization
  

  
To get a good feeling of how the model is performing and to compare with literature, compute the Pearson and Spearman correlations.
  
You can also plot the predictions in a scatterplot. We have added some code for that.   
