

# import pytorch

import torch

import torch.nn as nn

from torch import Tensor

from torch import optim

from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import lightning as L



# import basic functionality

import random

import numpy as np

import pandas as pd

import itertools



# libraries for plotting

import seaborn as sns

import matplotlib.pyplot as plt



import Bio

from Bio import SeqIO

torch.set_default_dtype(torch.float16)



# download and unpack DNA coding sequences for human, mouse and yeast

############################







# function that loads and processes a FASTA file containing coding sequences

def load_species_cds(file_name):

    dna_seq = []

    prot_seq = []

    for record in SeqIO.parse(file_name, "fasta"):

        # ensure that sequences are protein coding

        if 'gene_biotype:protein_coding' in record.description:

            if 'transcript_biotype:protein_coding' in record.description:

                if ' cds ' in record.description:

                    if len(record.seq) % 3 == 0:

                        dna_seq.append(str(record.seq))

                        prot_seq.append(str(record.seq.translate()))

                        

    # keep sequences that are protein coding

    dna_seq_cod = []

    prot_seq_cod = []

    for i in range(len(prot_seq)):

        if (prot_seq[i][0]=='M') & (prot_seq[i][-1]=='*'):

            dna_seq_cod.append(dna_seq[i])

            prot_seq_cod.append(prot_seq[i])



    # avoid sequences with undetermined/uncertain nucleotides

    dna_seq_cod = [dna_seq_cod[i] for i in range(len(dna_seq_cod)) if ('N' not in dna_seq_cod[i])]

    prot_seq_cod = [prot_seq_cod[i] for i in range(len(dna_seq_cod)) if ('N' not in dna_seq_cod[i])]

 

    # remove duplicates and randomly mix the list of sequences

    seqs = list(zip(dna_seq_cod, prot_seq_cod))

    seqs = list(set(seqs))

    random.shuffle(seqs)

    dna_seq_cod, prot_seq_cod = zip(*seqs)



    # pack samples as a list of dictionaries and return result

    seq_data = [{'dna':dna_seq_cod[i],'prot':prot_seq_cod[i]} for i in range(len(dna_seq_cod))]

    return seq_data

    


# load coding sequences for different species

print('loading human proteins')

seq_data = load_species_cds("all_seqs/Homo_sapiens.GRCh38.cds.all.fa")



# take a look at some sequences

[seq_data[i]['dna'][0:40]+'...' for i in range(5)]
print('number of sequences: ', len(seq_data))


# class to store a language

class Language:

    # initialize the language, as standard we have start of sentence (SOS), end of sentence (EOS) and a padding to equalize sentence lengths (PAD)

    def __init__(self, name, codon_len):

        self.name = name

        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}

        self.encoding = {}

        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>"}

        self.n_words = 3  # Count SOS and EOS

        self.codon_length = codon_len

        self.max_len = 0



    # function to add sentence to language (add new words in the sentence to our language)

    def addSentence(self, sentence):

        if len(sentence) > self.max_len:

            self.max_len = len(sentence)

        for word in [sentence[i:i+self.codon_length] for i in range(0, len(sentence), self.codon_length)]:

            self.addWord(word)



    # function to add word to language

    def addWord(self, word):

        if word not in self.word2index:

            self.word2index[word] = self.n_words

            self.index2word[self.n_words] = word

            self.n_words += 1

            

    # function to convert indices to one-hot encodings (i.e., 3 becomes [0,0,0,1,0,0,...])

    def as_one_hot(self):

        for key in self.word2index:

            new_val = np.zeros(len(self.word2index),dtype=np.int32)

            new_val[self.word2index[key]] = 1

            self.encoding[key] = new_val

    

    # function to convert indices to simple encodings

    def as_encoding(self):

        self.encoding = self.word2index



    # function to encode (and pad) a sentence

    # we use this to take an input sentence and convert it to a sequence of arrays that represent that sentence in a given language

    # in the context of proteins, think of this as encoding the bases or codons

    def encode(self, sentence, max_len=None):

        sos = [self.encoding["<SOS>"]]

        eos = [self.encoding["<EOS>"]]

        pad = [self.encoding["<PAD>"]]



        # split sentence in blocks of a given codon_length

        sentence_split = [sentence[i:i+self.codon_length] for i in range(0, len(sentence), self.codon_length)]

            

        # encode sentence in the given language

        sentence_encoded = [self.encoding[word] for word in sentence_split]

        max_len = round(max_len if max_len is not None else self.max_len / self.codon_length)

        # only pad or truncate if a maximum length is specified

        if max_len is not None:

            if len(sentence_split) < max_len - 2: 

                # sentence is shorter than max length-2; add SOS and EOS and pad to maximum length

                n_pads = max_len - 2 - len(sentence_split)

                return torch.Tensor(np.array(sos + sentence_encoded + eos + pad * n_pads, dtype = np.float16))

            else: 

                # sentence is longer than max length; truncate and add SOS and EOS

                sentence_truncated = sentence_encoded[:max_len - 2]

                return torch.Tensor(np.array(sos + sentence_truncated + eos, dtype = np.float16))

        else:

            return torch.Tensor(np.array(sos + sentence_encoded + eos))

########################

### step2a: create languages for DNA and protein sequences

########################



# create a language for DNA and protein sequences

dna_lang = Language(name="dna", codon_len=3)

prot_lang = Language(name="prot", codon_len=1)



# split the sequence data ('seq_data') that we defined above into sensible training, validation and test sets

# think about how much data would realistically be necessary to learn the problem of translating DNA sequences

train_set, val_set, test_set = torch.utils.data.random_split(seq_data, [0.7, 0.15, 0.15])



# memorize the dna and protein languages by parsing all sequences

for cur_seq in train_set:

    dna_lang.addSentence(cur_seq['dna'])

    prot_lang.addSentence(cur_seq['prot'])



# create an one-hot-encoding for all words codons and a simple encoding for all amino acids

# call the appropriate functions for each of the two languages

dna_lang.as_one_hot()

prot_lang.as_one_hot()



print(dna_lang.max_len)

print(prot_lang.max_len)



# here we define a function for encoding a dataset of dna and protein sequences

def encode_dataset(dataset, dna_lang, prot_lang, max_length):

    dataset_encoded = [

        { 

          'dna'  : dna_lang.encode(dataset[i]['dna'], max_len=max_length),

          'prot' : prot_lang.encode(dataset[i]['prot'], max_len=max_length)

        } for i in range(len(dataset))

    ]

    return dataset_encoded

    
########################

### step2b: encode your sequences here

########################



# define maximum number of codons

# we truncate any sequence longer than this length, and pad any sequence shorter than this length

# think about a sensible length for the input sequences

max_length = 42



# encode the training and validation data

train_set_encoded = encode_dataset(train_set, dna_lang, prot_lang, max_length)

val_set_encoded = encode_dataset(val_set, dna_lang, prot_lang, max_length)

print(train_set_encoded[0]['dna'].shape)

# take a look at the encoding of a DNA

train_set[0]['dna'], train_set_encoded[0]['dna']

print(train_set_encoded[0]['prot'].shape)

# take a look at the encoding of a protein

train_set[0]['prot'], train_set_encoded[0]['prot']



# define dataloader for the encoded sequences

def get_dataloader(dataset, batch_size):

    cur_sampler = RandomSampler(dataset)

    cur_dataloader = DataLoader(dataset, sampler=cur_sampler, batch_size=batch_size, drop_last=True, num_workers=15)

    return cur_dataloader

    
########################

### step 2c: create a dataloader for the validation and training sequences

########################



# how many samples should be trained on simultaneously?

batch_size = 20



# define dataloader for training

train_loader = get_dataloader(train_set_encoded, batch_size)

val_loader = get_dataloader(val_set_encoded, batch_size)



# Define the device (CPU or GPU)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################

### step 3a: define the model architecture

########################

class MyLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super(MyLSTM,self).__init__()



        self.l1 = nn.LSTM(input_size, hidden_size, batch_first=True)

        self.act =  nn.Identity()#nn.ReLU()

        self.l2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.l3 = nn.LSTM(hidden_size, output_size, batch_first=True)



    def forward(self,inp):

        #print(inp.shape)

        inp.to(device)

        output, (hn2, cn2) = self.l2(self.act(output))

        output, (hn3, cn3) = self.l3(self.act(output))

        return output #nn.Softmax()(output)

        
########################

### step 3b: define the lightning module to train the model

########################



# lightning module to train the sequence model

class SequenceModelLightning(L.LightningModule):

    def __init__(self, input_size, hidden_size, output_size, lr=0.1):

        super().__init__()

        self.model = MyLSTM(input_size, hidden_size, output_size).double()

        self.lr = lr

        # define loss function here, pseudocode:

        self.loss = nn.CrossEntropyLoss()



    def forward(self, x):

        return self.model(x)

    

    def training_step(self, batch, batch_idx):

        input_tensor = batch["dna"].double()

        target_tensor = batch["prot"].double()



        output = self.model(input_tensor)

        loss = self.loss(output,target_tensor)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    

    def validation_step(self, batch, batch_idx):

        input_tensor = batch["dna"].double()

        target_tensor = batch["prot"].double()



        output = self.model(input_tensor)

        #print(output.shape, target_tensor.shape)

        #print(output.view(-1, output.shape[2]).shape, target_tensor.view(-1).long().shape)

        loss = self.loss(output,target_tensor)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    

    def configure_optimizers(self):

        # define optimizer here

        return torch.optim.Adam(self.model.parameters(), self.lr)

    
########################

### step3c: define the input parameters for the training loop

########################



# define the model and training loop

# think of the dimensionality of your input data (dna sequences) and output data (protein sequence), and where these numbers are stored

lit_model = SequenceModelLightning(input_size = 67,

                                  hidden_size = 200,

                                  output_size = 24,

                                  lr = 8e-3)



# define the trainer

trainer = L.Trainer(devices = 1, 

                    max_epochs = 15)



# learn the weights of the model

print("traininig")

trainer.fit(lit_model, train_loader, val_loader)

print("done")
print("done")


# evaluate the performance of your model on the validation data

trainer.validate(lit_model, val_loader)



# show the model architecture

my_lstm = lit_model.model

print("my_lstm
 = ", my_lstm
)


# we encode the test data using the same dna and protein language encodings we defined before

# if you change the languages, you need to re-encode the test sequences as well!

test_set_encoded = encode_dataset(test_set, dna_lang, prot_lang, max_length)

########################

### step4: define the input tensor and get the prediction from your model

########################



# pick a random sequence from the test set

random_pair = np.random.randint(0,len(test_set))



# get the encoded dna sequence and its known protein translation

dna_sequence = np.array([test_set_encoded[random_pair]['dna']])

protein_translation = test_set[random_pair]['prot']



# send model and input sequence to device, compute translation of sequence

my_lstm.cuda()

input_tensor = torch.from_numpy(dna_sequence).to(device).double()

output = my_lstm(input_tensor)

print(output.shape)

# convert output back to protein sequence by taking the most likely amino acid per position, print results

result = "".join([prot_lang.index2word[i] for i in output.cpu().topk(1)[1].view(-1).numpy()])



print(len(result))

print('     '+protein_translation)

print(result, end='\n\n')



# print accuracy

result = "".join([prot_lang.index2word[i] for i in output.cpu().topk(1)[1].view(-1).numpy() if i not in [key for key in Language('',1).index2word]])

min_len = np.min([len(result),len(protein_translation)])

print('Accuracy of aa calling over the sequence: ', np.sum([protein_translation[i] == result[i] for i in range(min_len)])/min_len)

