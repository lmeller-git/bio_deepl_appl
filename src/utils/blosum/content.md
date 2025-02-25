# Project: Protein stability prediction
  

  
In the project you will try to predict protein stability changes upon point mutations. 
  
We will use acuumulated data from experimental databases, i.e. the Megascale dataset. The other notebooks contain a more detailed description. 
  

  
Here we compute a simple baseline with the BLOSUM substitution matrix.  
# Imports  
## Dataloading
  

  
We are using the Megascale dataset. The train, validation and test sets are already predefined.
  
The use of the dataloaders is explained in more detail in the other notebooks.  
## Testing
  

  
To get a good feeling of how the model is performing and to compare with literature, compute the Pearson and Spearman correlations and the RMSE (root mean square error).  
