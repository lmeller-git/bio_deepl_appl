# Lab 3: RNNs and LSTMs
  
In this tutorial we create neural networks using [Long Short-Term Memory (LSTM)](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) cells. The data and its pre-processing for of this notebook is identical to the first lab (Perceptron) to keep the amount of new information limited. Some of the required code-blocks are empty - requiring your imput to complete the model. A few additional questions at the end challenge you to play around with the code and try things for yourselves.
  

  
During the session, you will create a LSTM to translate DNA sequences into protein sequences.   
# Step 1: Pre-processing the data
  
Here we download and pre-process the dataset. As before, we only consider DNA sequences that are protein coding, contain a integer number of codons, have a start and stop codon, and do not contain any uncertain nucleotides. Finally, we remove duplicates and randomly mix the sequences. Nothing is different from the last time, so you can simply execute these steps and move on to Step 2.  
# Step 2: Encoding the sequences
  
Having prepared the coding sequences and their translation, we convert them into a numeric representation as vectors. To do so, we first construct a language class that stores words of each language and allows us to convert between encoding/indices and words in a language. We define a function that allows us to store any sequence of words (i.e., codons or bases) as a numeric representation. Here we extend every sequence with a start of sentence <SOS> and end of sentence <EOS> token, such that the model knows when to start and stop translating. Finally, we can extend every sentence to the same length by padding with an empty "word" that is not translated or used, but allows us to use the identical-length numerical representation of the sentences as input for the model. The language allows for a simple encoding of words as numbers or as numerical vectors (one-hot-encoding). The 'encode' function converts an input sentence to the specified encoding.  
# Step 3: Define model
  
We created languages for DNA and protein sequences, and encoded all sequences through the encodings defined by these languages. We then created data loaders for these encoded sequences. As a final preparation, we define our [model](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html). 
  
Create a class that instructs pytorch to make a LSTM model using the given input, hidden size and output parameters. You'll have to define the init and forward functions. Additionally, in the Pytorch Lightning class, you must choose a loss function and optimizer that are appropriate for the problem you are trying to solve. Once you have thought about your model, we will go over the architecture together with the class. Hint: for the LSTM, you'll need to also define the hidden and cell states.  
# Step 4: Test the model on random test sequences  
# Steps: <br>
  
2a: create languages for the DNA and protein sequences <br>
  
2b: encode the training and validation sequences <br>
  
2c: create a dataloader for the validation and training sequences <br>
  
3a: define your LSTM model <br>
  
3b: define the lightning module to train the model <br>
  
3c: define the input parameters for the training loop <br>
  
Train your model :) <br>
  
4: define the input tensor and get the prediction from your model <br>
  
 <br>
  
# Questions: <br>
  
-The test sequences are truncated at or padded to max_length. Change the encoding of the test sequences to work for arbitrary lengths (the actual length of the sequences). <br>
  
-The model is trained on truncated and padded sequences. Change the setup to train your model on arbitrary length sequences (their actual length). Before you train the model, think about a) the number of samples to use for training, b) batch sizes, c) number of epochs <br>
  
-Can you modify the code to train it on a single DNA sequence (i.e. a single sample)? Can you achieve perfect/reasonable accuracy? <br>
  
-What is the minimum model size to reach a 'good' accuracy? What is the minimum number of required samples? Think about these questions in context of a classical machine learning classifier <br>
  
-Can you change the code to use a DNA language for single nucleotides instead of codons and train the model?  
