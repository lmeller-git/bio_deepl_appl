import numpy as np, pandas as pd, sklearn.preprocessing

import datasets, evaluate, transformers # Hugging Face libraries https://doi.org/10.18653/v1/2020.emnlp-demos.6

import Bio.SeqIO.FastaIO # Biopython for reading fasta files

from IPython.display import display, HTML

random_number = 4 # https://xkcd.com/221/
# Uncomment & execute once to download data from https://services.healthtech.dtu.dk/services/DeepLocPro-1.0/

#!mkdir -p data

#!curl https://services.healthtech.dtu.dk/services/DeepLocPro-1.0/data/graphpart_set.fasta -o data/graphpart_set.fasta
with open('data/graphpart_set.fasta') as handle:

    fasta_cols = ['header', 'sequence']

    df_data = pd.DataFrame.from_records([values for values in Bio.SeqIO.FastaIO.SimpleFastaParser(handle)], columns=fasta_cols)

header_cols = ['uniprot_id', 'subcellular_location', 'organism_group', 'fold_id']

df_data[header_cols] = df_data['header'].str.split('|', expand=True)

final_cols = ['uniprot_id']

df_data = df_data[['uniprot_id', 'subcellular_location', 'organism_group', 'fold_id', 'sequence']].astype({'fold_id': int}).sort_values('fold_id')

print("df_data = ", df_data)
# Encode subcellular location as numerical labels

subcellular_location_encoder = sklearn.preprocessing.LabelEncoder()

subcellular_location_encoder.fit(df_data['subcellular_location'])

df_data['label'] = subcellular_location_encoder.transform(df_data['subcellular_location'])

print("df_data = ", df_data)
train_id = {0, 1, 2}

eval_id = {3}

test_id = {4}



df_train = df_data.query('fold_id in @train_id')#.groupby('subcellular_location').sample(n=50, random_state=random_number)

df_eval = df_data.query('fold_id in @eval_id')

df_test = df_data.query('fold_id in @test_id')

print(len(df_train), 'records in training data:')

print(df_train['subcellular_location'].value_counts())

print()

print(len(df_eval), 'records in eval data:')

print(df_eval['subcellular_location'].value_counts())

print()

print(len(df_test), 'records in test data:')

print(df_test['subcellular_location'].value_counts())
# Prepare train/eval/test data sets for ESM2 model

model_checkpoint = 'facebook/esm2_t6_8M_UR50D'

tokenizer = transformers.AutoTokenizer.from_pretrained(model_checkpoint)



train_tokenized = tokenizer(df_train['sequence'].tolist(), truncation=True, max_length=1024)

eval_tokenized = tokenizer(df_eval['sequence'].tolist(), truncation=True, max_length=1024)

test_tokenized = tokenizer(df_test['sequence'].tolist(), truncation=True, max_length=1024)



train_dataset = datasets.Dataset.from_dict(train_tokenized).add_column('labels', df_train['label'].tolist())

eval_dataset = datasets.Dataset.from_dict(eval_tokenized).add_column('labels', df_eval['label'].tolist())

test_dataset = datasets.Dataset.from_dict(test_tokenized).add_column('labels', df_test['label'].tolist())
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=df_data['label'].nunique())

print("model = ", model)
# Track accuracy and macro F1 score throughout the training

# https://huggingface.co/docs/transformers/en/training#evaluate

metric_accuracy = evaluate.load('accuracy')

metric_f1 = evaluate.load('f1')



def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=1)

    return {

        'accuracy': metric_accuracy.compute(predictions=predictions, references=labels)['accuracy'],

        'f1_macro': metric_f1.compute(predictions=predictions, references=labels, average='macro')['f1'],

    }
# Set up fine-tuning

trainer_args = transformers.TrainingArguments(

    output_dir=f'{model_checkpoint}-subcellular_location',

    eval_strategy='epoch',

    save_strategy='epoch',

    load_best_model_at_end=True,

    metric_for_best_model='accuracy',

    # Adjust if needed

    #per_device_train_batch_size,

    #per_device_eval_batch_size,

)



trainer = transformers.Trainer(

    model,

    trainer_args,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset,

    tokenizer=tokenizer,

    compute_metrics=compute_metrics,

)



retrained = trainer.train()
# Use fine-tuned model to predict on held-out test data

retrained_predict = trainer.predict(test_dataset=test_dataset)

retrained_predict.metrics
# Convert probabilities into discrete predictions by taking the max probability

test_labels = np.argmax(retrained_predict.predictions, axis=-1)

# Sanity-check by manualy calculating the accuracy

print(sum(test_labels == test_dataset['labels']) / len(test_dataset))

print("test_labels = ", test_labels)
fold_id = set(df_data.fold_id)

predicted_labels = []

for test_id in sorted(fold_id):

    eval_id = (test_id + 1) % 5

    train_id = fold_id - set([eval_id, test_id])



    df_train = df_data.query('fold_id in @train_id')#.groupby('subcellular_location').sample(n=10, random_state=random_number)

    df_eval = df_data.query('fold_id == @eval_id')

    df_test = df_data.query('fold_id == @test_id')

    print(train_id, eval_id, test_id, len(df_train), len(df_eval), len(df_test))



    # ...



    # Predict labels, and gather the predictions for the held-out test data into predicted_labels

    retrained_predict = trainer.predict(test_dataset=test_dataset)

    predicted_labels += list(np.argmax(retrained_predict.predictions, axis=-1))
# Show table with performance metrics split by organism to match Table 2

def calculate_stats_(df):

    accuracy = metric_accuracy.compute(predictions=df.predicted_labels.values, references=df.label.values)['accuracy']

    f1_macro = metric_f1.compute(predictions=df.predicted_labels.values, references=df.label.values, average='macro')['f1']

    return pd.Series({

        'size': '{:d}'.format(len(df)),

        'accuracy': '{:.2f}'.format(accuracy),

        'f1_macro': '{:.2f}'.format(f1_macro),

    })



df_data['predicted_labels'] = predicted_labels

pd.concat([

    calculate_stats_(df_data),

    calculate_stats_(df_data.query('organism_group == "archaea"')),

    calculate_stats_(df_data.query('organism_group == "positive"')),

    calculate_stats_(df_data.query('organism_group == "negative"')),

], axis=1).set_axis(['Overall', 'Archaea' , 'Gram pos', 'Gram neg'], axis=1)
