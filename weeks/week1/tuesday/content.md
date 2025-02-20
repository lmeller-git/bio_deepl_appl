# Lab 1: Introduction to Neural Networks
  
In this tutorial we introduce some of the concepts for working with neural networks using [Pytorch](https://pytorch.org/tutorials/recipes/recipes_index.html). The entire notebook can be executed as-is, given the lack of time for this first lab session. We encourage you to explore the code yourselves to get comfortable with the concepts of deel learning in the context of biology. A few questions at the end challenge you to play around with the code and try things for yourselves.
  

  
In this session, you will create a simple neural network that classifies any given DNA sequence as protein coding or not. As a starting point, we use as examples the coding DNA sequences from humans (homo sapiens (HS)). As negatives, we use random sequences of DNA where each nucleotide is drawn from a uniform distribution over the possible nucleotides. We then train a neural network on de [codon frequencies](https://en.wikipedia.org/wiki/DNA_and_RNA_codon_tables) of these sequences.
  

  
In addition to human DNA sequences, we also take a look at coding sequences from mice ([mus musculus (MM)](https://en.wikipedia.org/wiki/House_mouse)) and yeast ([saccharomyces cerevisiae (SC)](https://en.wikipedia.org/wiki/Saccharomyces_cerevisiae)). There are subtle differences between the coding frequencies of these species. You will test how well your human-trained model is able to recover the coding sequences for mice and yeast (think of your results in the context of evolutionary distances between species). 
  
# Step 1: Pre-processing the data
  
Here we download and pre-process the dataset. We only consider DNA sequences that are protein coding, contain a integer number of codons, and have a start and stop codon. Finally, we remove duplicates and randomly mix the sequences.   
# Step 2: Encoding sequences as codon frequencies
  
The steps above give us a set of unique coding sequences for humans, mice and yeast. To train a neural network on de coding frequencies of these sequences, we encode the sequences by converting each sequence to an array of frequencies for each possible codon. Each codon gets assigned a index in the array. We first create a 'language' that knows all possible words (codons) for a given codon length and input sequence. This language then allows us to convert between codon (e.g., 'ATG') and indices in the array (e.g., 'ATG' -> 0) to keep track of the codon frequencies. Here, we use Tensors - the datatype used for pytorch data - to store the coding frequencies.
  

  
Biologically, [DNA codons](https://en.wikipedia.org/wiki/DNA_and_RNA_codon_tables) consist of three nucleotides, encoding amino acids. However, since we are training a neural network to classify a sequence to be protein coding or not, we can choose any number of nucleotides to represent a 'codon'. For example, we can choose a "codon length" (codon_length) of a single nucleotide (which would result in us training the model on the [frequencies of nucleotides in DNA](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC403801/)), or a codon length of two nucleotides (no biological meaning as this does not represent a biological unit - we do not expect a model to learn any biology at all), or a codon length of 6 nucleotides (representing pairs of amino acids - would this yield a model that "learns" any biology?). You can play around with the codon_length yourself, but we start with a codon length of 3 nucleotides - represeting one amino acid.   
After creating a language to convert between DNA sequences and coding frequencies, we split our data into training, validation and test sets of appropriate sizes, and we encoded our DNA sequences into codon frequencies. Finally, for each sequence, we also created a random sequence of codons following the same sequence length distribution as the DNA sequences. These nucleotides are drawn from a random uniform distribution over the possible codons. These random sequences are used as negatives (i.e., label = 1 will tell the model that a sequence is protein coding, label = 0 tells the model that a sequence is not protein coding).   
# Step 3: Creating a dataloader
  
Having encoded our DNA sequences as codon frequencies, we are ready to prepare the data for training a neural network. We will create a 'dataloader' that takes care of splitting the data into batches.  
# Step 4: Define model
  
As a final preparation, we define our [model](https://pytorch.org/docs/stable/generated/torch.nn.Module.html). We create a class that instructs pytorch to follow a certain architecture for the model. Our model initializes all relevant parts (init function) and tells pytorch how to compute the output for a given input (forward function). For our perceptron, we use a single dense linear layer with a sigmoidal activation function. You can play around with the model architecture later - several options are left as comments.
  
We are trying to train a model for solving a classfication problem. The labels for our samples are binary (ones and zeros). We therefore use a binary loss function.   
# Step 5: Train simple model 
  
To train our model, we need a "training loop". 
  
1) First, we tell pytorch that we want to train our model (so it has to keep track of gradients). 
  
2) We then iterate over our data in batches to speed up computations (there is little advantage for computing the gradient with all samples over, say, a few hundred samples). 
  
3) We set the gradients to zero (we don't want to re-use previous computations for our next training step).
  
4) We compute the output of the model for the given input sequences. 
  
5) We then compute the loss of the model output for the given target labels of the input sequences and [backpropagate](https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html) the loss through the network to compute the gradient. 
  
6) Finally, we instruct the optimizer to use the gradient and perform one appropriately-sized [step](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html) to update the model weights. 
  
7) To keep track of our effors we compute the accuracy and training loss for the current samples.
  

  
Finally, we train our model using the given data and training loop  
Similar to the training loop, we need a "test loop" to get the output of the model for a given set of validation samples on which we do not train the model. 
  
1) First, we tell pytorch that we do NOT want to train our model (no keeping track of gradients - evaluation mode). 
  
2) We then iterate over our validation data. 
  
3) We compute the output of the model for the given input sequences. 
  
4) We then compute the loss of the model output for the given target labels of the input sequences.
  
5) We compute the accuracy and training loss for the current samples.  
# Step 7: Plotting error and accuracy
  
Using the output from training, we can plot the results for each epoch to look at the learning of our model. For this, we average the errors over the different folds from training. Carefully look at the training and testing error to choose an appropriate number of epochs for training (to avoid overfitting). 
  

  
NOTE: the initial model included trains very fast with a small error and high accuracy. The error and accuracy become interesting when looking at training on other species as negative samples in the questions.  
# Step 8: Questions
  
1) how many parameters does your model have?
  
2) change model architecture (e.g., more layers, other parameters, dropout parameters)
  
3) adapt the code to compute the probabilities for mouse and yeast DNA sequences. Having trained the model on human DNA sequences, what is the difference in average probabilities between species? Why?
  
4) train the model using mouse or yeast sequences as negative samples for coding sequences. This creates a model that learns to classify sequences as being likely from mice/yeast or from humans
  
5) what happens if you change the codon length?
  
6) what happens to the probablities when sequences are frameshifted? can you train the model using frameshifted sequences as input?
  
7) train the model on amino acid frequencies instead of codons  
