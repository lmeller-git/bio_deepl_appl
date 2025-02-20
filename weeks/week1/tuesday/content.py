

# import pytorch

import torch

import torch.nn as nn

from torch import Tensor

from torch import optim

import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from sklearn.model_selection import KFold



# import basic functionality

import random

import numpy as np

import pandas as pd



# libraries for plotting

import seaborn as sns

import matplotlib.pyplot as plt





import Bio

from Bio import SeqIO


# download and unpack DNA coding sequences for human, mouse and yeast

############################



!mkdir -p ~/all_seqs

%cd ~/



!wget -P ~/all_seqs/ https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/cds/Homo_sapiens.GRCh38.cds.all.fa.gz

!gzip -df "all_seqs/Homo_sapiens.GRCh38.cds.all.fa.gz"



!wget -P ~/all_seqs/ https://ftp.ensembl.org/pub/current_fasta/saccharomyces_cerevisiae/cds/Saccharomyces_cerevisiae.R64-1-1.cds.all.fa.gz

!gzip -df "all_seqs/Saccharomyces_cerevisiae.R64-1-1.cds.all.fa.gz"



!wget -P ~/all_seqs/ https://ftp.ensembl.org/pub/current_fasta/mus_musculus/cds/Mus_musculus.GRCm39.cds.all.fa.gz

!gzip -df "all_seqs/Mus_musculus.GRCm39.cds.all.fa.gz"



############################

### some unnecessarily complex code to process the FASTA files for coding sequences

############################



# function that loads and processes a FASTA file containing coding sequences

def load_species_cds(file_name, max_nr_samples):

    seqs = []

    for record in SeqIO.parse(file_name, "fasta"):

        # ensure that sequences are protein coding

        if 'gene_biotype:protein_coding' in record.description:

            if 'transcript_biotype:protein_coding' in record.description:

                if ' cds ' in record.description:

                    if len(record.seq) % 3 == 0:

                        # translate sequence and check for start and stop codons

                        code_translation = str(record.seq.translate())

                        if (code_translation[0]=='M') & (code_translation[-1]=='*'):

                            seqs.append(str(record.seq))



    # avoid sequences with undetermined/uncertain nucleotides

    # restrict to sequences with at least 100 aa for codon frequency estimation

    seqs = [seqs[i] for i in range(len(seqs)) if (len(seqs[i])>=300)]#('N' not in train_cds_filtered[i]) and (len(train_cds_filtered[i])>=300)]

    

    # remove duplicates and randomly mix the list of sequences

    seqs = list(set(seqs))

    random.shuffle(seqs)

    

    return list(seqs)[0:max_nr_samples]



# there are many sequences, given the time constraints, we limit the number of sequences to speed up the processing

max_nr_samples = 20000



# load coding sequences for different species

print('loading human proteins')

seq_data_human = load_species_cds("all_seqs/Homo_sapiens.GRCh38.cds.all.fa", max_nr_samples)



print('loading yeast proteins')

seq_data_yeast = load_species_cds("all_seqs/Saccharomyces_cerevisiae.R64-1-1.cds.all.fa", max_nr_samples)



print('loading mouse proteins')

seq_data_mouse = load_species_cds("all_seqs/Mus_musculus.GRCm39.cds.all.fa", max_nr_samples)



# take a look at some sequences

[seq_data_human[i][0:20]+'...'+seq_data_human[i][-20:] for i in range(5)]



# class to store a language

class Language:

    # initialize the language, as standard we have start of sentence (SOS), end of sentence (EOS) and a padding to equalize sentence lengths (PAD)

    def __init__(self, name, codon_len):

        self.name = name

        self.word2index = {}

        self.index2word = {}

        self.n_words = 0

        self.codon_length = codon_len





    # function to split sentence in blocks of a given codon_length

    def splitSentence(self, sentence):

        return [sentence[i:i+self.codon_length] for i in range(0,len(sentence),self.codon_length) if len(sentence[i:i+self.codon_length])==self.codon_length]





    # function to add sentence to language (add all new words in the sentence to our language)

    def learnWords(self, sentence):

        sentence_split = self.splitSentence(sentence)

        for word in sentence_split:

            if word not in self.word2index:

                self.word2index[word] = self.n_words

                self.index2word[self.n_words] = word

                self.n_words += 1





    # function to count the word frequencies in a sentence

    def encode(self, sentence):

        sentence_split = self.splitSentence(sentence)

    

        ############################

        ### create a tensor having the number of available words as length, fill with with zeros

        codon_freqs = torch.zeros(self.n_words)

        ############################



        # count frequencies of every word in the sentence

        word_counts = np.unique(sentence_split, return_counts = True)

        for i in range(len(word_counts[0])):

            codon_freqs[self.word2index[word_counts[0][i]]] = word_counts[1][i]

        codon_freqs /= len(sentence_split)



        return codon_freqs





    # return a sample of frequencies for all words in a language

    # here we're matching the codon frequencies to the sequence length of the provided sequence

    def sample_sentence(self, sentence):

        # create a tensor having the number of available words as length, fill with with zeros

        codon_freqs = torch.zeros(self.n_words)



        # sample nr of codons in sequence based on actual data

        # generate random sequence of codons given the current sequence length

        nr_codons = round(len(sentence)/self.codon_length)

        sampled_codons = list(np.random.randint(low=0, high=self.n_words, size = nr_codons, dtype=int))



        # count frequencies of every codon

        word_counts = np.unique(sampled_codons, return_counts = True)

        for i in range(len(word_counts[0])):

            codon_freqs[word_counts[0][i]] = word_counts[1][i]

        codon_freqs /= nr_codons



        return codon_freqs





    # here we define a function for encoding an entire dataset sequences

    def encode_dataset(self, dataset):

        # encode positives

        encoded_positives = [{'sample':self.encode(sentence),'label':torch.Tensor([1])} for sentence in dataset]



        # sample negatives following the sequence length distribution of positives

        encoded_negatives = [{'sample':self.sample_sentence(sentence),'label':torch.Tensor([0])} for sentence in dataset]



        # merge datasets and randomly mix positives and negatives

        dataset_encoded = encoded_positives + encoded_negatives

        random.shuffle(dataset_encoded)

        

        return dataset_encoded





codon_length = 3



# create a language for human DNA sequences

dna_lang = Language(name="dna_human", codon_len=codon_length)



# memorize the dna language by parsing all sequences

for cur_seq in seq_data_human:

    dna_lang.learnWords(cur_seq)



# split the sequence data ('seq_data_human') that we defined above into training, validation and test sets

train_set_human, val_set_human, test_set_human = torch.utils.data.random_split(seq_data_human, [0.5,0.4,0.1])



# encode the training validation and test data

train_set_human_encoded = dna_lang.encode_dataset(train_set_human)

val_set_human_encoded = dna_lang.encode_dataset(val_set_human)

test_set_human_encoded = dna_lang.encode_dataset(test_set_human)

# take a look at one of the samples:

train_set_human_encoded[0]


# as before but for the other species:

# split the sequence data that we defined above into training, validation and test sets

# encode the training and validation data



train_set_mouse, val_set_mouse, test_set_mouse = torch.utils.data.random_split(seq_data_mouse, [0.5,0.4,0.1])

train_set_mouse_encoded = dna_lang.encode_dataset(train_set_mouse)

val_set_mouse_encoded = dna_lang.encode_dataset(val_set_mouse)

test_set_mouse_encoded = dna_lang.encode_dataset(test_set_mouse)



train_set_yeast, val_set_yeast, test_set_yeast = torch.utils.data.random_split(seq_data_yeast, [0.5,0.4,0.1])

train_set_yeast_encoded = dna_lang.encode_dataset(train_set_yeast)

val_set_yeast_encoded = dna_lang.encode_dataset(val_set_yeast)

test_set_yeast_encoded = dna_lang.encode_dataset(test_set_yeast)



# explore the correlation of coding frequencies across species

# first average and merge codon frequencies of the different species and random sequences

codon_freqs = pd.DataFrame([np.mean(np.array([ch['sample'] for ch in train_set_human_encoded if ch['label']==1]),axis=0),

                         np.mean(np.array([ch['sample'] for ch in train_set_mouse_encoded if ch['label']==1]),axis=0),

                         np.mean(np.array([ch['sample'] for ch in train_set_yeast_encoded if ch['label']==1]),axis=0),

                         np.mean(np.array([ch['sample'] for ch in train_set_human_encoded if ch['label']==0]),axis=0)

                         ]).T



# label codons and sort by human frequency

codon_freqs.index = [dna_lang.index2word[idx] for idx in list(codon_freqs.index)]

codon_freqs.reset_index(inplace=True)

codon_freqs.columns = ['codon','human','mouse','yeast','random']

codon_freqs.sort_values(by='human',ascending=False,inplace=True)



print('correlation matrix: ')

print(codon_freqs.set_index('codon').corr())



# plot coding frequencies for different species

# initialize figure

sns.set(rc={'figure.figsize':(5,15)})

sns.set(font="Arial")

sns.set(style="whitegrid")



# stack dataframe of frequencies

plot_freqs = codon_freqs.set_index('codon').stack().reset_index()

plot_freqs.columns = ['codon','origin','frequency']



# plot codon frequencies for different species

sns.barplot(data=plot_freqs,x='frequency',y='codon',hue='origin')

plt.tight_layout()

plt.show()



############################

# define a function to create a dataloader for the encoded sequences

def get_dataloader(dataset, batch_size):

    cur_sampler = RandomSampler(dataset)

    cur_dataloader = DataLoader(dataset=dataset, sampler=cur_sampler, batch_size=batch_size, drop_last=True, num_workers=15)

    return cur_dataloader    

############################



# how many samples should be trained on simultaneously?

batch_size = 300



# define dataloader for training

train_loader_human = get_dataloader(train_set_human_encoded, batch_size)

val_loader_human = get_dataloader(val_set_human_encoded, batch_size)

test_loader_human = get_dataloader(test_set_human_encoded, 1)



# Define the device (CPU or GPU)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device



# Define the model architecture

class myPerceptron(nn.Module):

    def __init__(self, input_param, hidden_param, output_param, dropout_prob):

        super(myPerceptron, self).__init__()

        

        self.input_param = input_param

        self.hidden_param = hidden_param

        self.output_param = output_param

        

#        self.dropout = nn.Dropout(dropout_prob)

        self.linear0 = nn.Linear(input_param, output_param)

        self.sigmoid = nn.Sigmoid() # or other activation function (e.g. ReLU)



    def forward(self, inp):

#        inp_drop = self.dropout(inp)

        layer0 = self.linear0(inp)

        output = self.sigmoid(layer0)

        return output



# use binary cross entropy los for this classification problem

my_loss_function = nn.BCELoss()



# initialize an instance of our model class (a variable that is a model following the architecture we defined above)

my_model = myPerceptron(dna_lang.n_words, # size of input tensors (the number of codons)

                     20, # size of a hidden layer (not used for now)

                     1, # size of the model's output

                     0 # some additional parameter that could be used (e.g. dropout frequency)

                    ).to(device) # send model to device



# show model architecture

my_model



# Define the training loop

def train(model, train_loader, optimizer, device):

    # training mode

    model.train(True)

    

    # Enabling gradient calculation

    with torch.set_grad_enabled(True):

        collect_loss = 0

        correct = 0

        nr_samples = 0

        for batch_idx, data in enumerate(train_loader):

            # send features and labels to GPU/CPU

            model_input = data['sample'].to(device)

            target = data['label'].to(device)



            # zero the gradients

            model.zero_grad()

            optimizer.zero_grad()



            # compute output of model

            output = model(model_input)



            # compute the loss and update model parameters

            loss = my_loss_function(output, target)

            loss.backward()



            # adjust learning weights

            optimizer.step()

            

            # store training loss

            collect_loss += loss.item()*batch_size

            

            # compute accuracy of training data

            pred = torch.round(output,decimals=0)

            correct += (pred.eq(target.view_as(pred)).sum().item())

            nr_samples += len(target)

            

        return {'train_loss':collect_loss/nr_samples, 'train_accuracy':correct/nr_samples}

    


# define the test loop

def validate(model, test_loader, device):

    # Evaluation mode

    model.eval()

    

    with torch.no_grad():

        collect_loss = 0

        correct = 0

        nr_samples = 0

        for data in test_loader:

            # send features and labels to GPU/CPU

            model_input = data['sample'].to(device)

            target = data['label'].to(device)

            

            # compute output of model

            output = model(model_input)



            # store test loss

            collect_loss += my_loss_function(output, target).item()*batch_size

            

            # compute accuracy for test data

            pred = torch.round(output,decimals=0)

            correct += (pred.eq(target.view_as(pred)).sum().item())

            nr_samples += len(target)

            

        return {'val_loss':collect_loss/nr_samples, 'val_accuracy':correct/nr_samples}

    


# define the number of epochs - how often should the model (my_model) see all of the data (train_loader_human)?

n_epochs = 20



# initialize an instance of our model class (a variable that is a model following the architecture we defined above)

my_model = myPerceptron(dna_lang.n_words, # size of input tensors (the number of codons)

                     20, # size of a hidden layer (not used for now)

                     1, # size of the model's output

                     0 # some additional parameter that could be used (e.g. dropout frequency)

                    ).to(device) # send model to device



# use stochastic gradient descent with the given learning rate

optimizer = torch.optim.Adam(my_model.parameters(), lr=0.01)



# Train the model on the current data

stats_tracker = []

for epoch in range(0, n_epochs):

    # train the model and get training loss

    test_stats = validate(my_model, val_loader_human, device)

    train_stats = train(my_model, train_loader_human, optimizer, device)

    stats_tracker.append( train_stats|test_stats )

    print('epoch: ', epoch, train_stats, test_stats, '\t\t\t\t\t\t\t\t', end='\r')

    


# initialize figure

sns.set(rc={'figure.figsize':(5,5)})

sns.set(font="Arial")

sns.set(style="whitegrid")



# format loss data

plot_data = pd.DataFrame(stats_tracker)

plot_data = plot_data.stack().reset_index()

plot_data.columns = ['epoch','dataset','value']



# plot training and test loss as function of epoch

ax=sns.lineplot(data=plot_data, x='epoch', y='value',hue='dataset')

#ax.set_yscale('log')

ax.set_ylim([0,1])

plt.tight_layout()

plt.show()



# define function for evaluating the trained model for a given sample

def evaluate(model, sample):

    # set the model in evaluation mode without computing gradients

    model.eval()

    with torch.no_grad():

        # compute the output of the model for a given sample

        output = my_model(sample.to(device))

    return output.item()





# finally, using a trained model, we can compute a 'probability' that a given input sequence is encoding a protein

test_sampler = enumerate(test_loader_human)



# here we pick a random sequence that was not used for training, but you can change this to any sequence you would like 

batch_idx, test_sample = next(test_sampler)



# evaluate model for given test sample

output = evaluate(my_model, test_sample['sample'])



# print output of the model together with the label of the sample

print('probability: ',output, test_sample['label'])

