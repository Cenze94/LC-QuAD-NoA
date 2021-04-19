from typing import List, Dict, Any, Tuple
import json

# Libraries

import matplotlib.pyplot as plt
import pandas as pd
import torch

# Preliminaries

from torchtext.data import Field, TabularDataset, BucketIterator, Iterator

# Models

import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

# Training

import torch.optim as optim

# Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Torchtext

from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

general_settings = {
    # Check whether cuda is available
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "SEED": 2021,
    "BATCH_SIZE": 32,
    "valid_data_size": 0.15,
    "destination_path": "output"
}

# Support function to avoid code repetition, get questions and labels lists and save temporary JSON file
def get_lists_from_elements(elements_list: List[Dict[str, Any]], filename: str):
    questions = []
    labels = []
    for element in elements_list:
        if element['question']:
            questions.append(element['question'])
        else:
            questions.append(element['NNQT_question'])
        if 'has_answer' in element and element['has_answer'] == False:
            labels.append(0)
        else:
            labels.append(1)
    with open(filename + "_temp.json", "w") as json_file:
        for i in range(len(questions)):
            # Remove initial '"' characters if present
            if questions[i][0] == '"' and questions[i][-1] == '"':
                questions[i] = questions[i][1:-1]
            json_file.write(json.dumps({"label": labels[i], "question": questions[i]}, ensure_ascii=False) + "\n")

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        options_name = "bert-base-cased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name, num_labels=2, output_attentions=False, output_hidden_states=False)
        self.encoder.cuda()

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, token_type_ids=None, labels=label)[:2]
        return loss, text_fea

# Save and Load Functions
def save_checkpoint(save_path, model, valid_loss):
    if save_path == None:
        return
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):
    if load_path==None:
        return
    state_dict = torch.load(load_path, map_location=general_settings['device'])
    print(f'Model loaded from <== {load_path}')
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']

def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path == None:
        return
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_metrics(load_path):
    if load_path==None:
        return
    state_dict = torch.load(load_path, map_location=general_settings['device'])
    print(f'Model loaded from <== {load_path}')
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

# Training Function
def train(model, optimizer, train_loader, valid_loader, criterion = nn.BCELoss(), num_epochs = 5,
          eval_every = None, file_path = general_settings['destination_path'], best_valid_loss = float("Inf")):
    # Initialize running values
    if eval_every is None:
        eval_every = len(train_iter) // 2
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for (labels, questions), _ in train_loader:
            labels = labels.type(torch.LongTensor)
            questions = questions.type(torch.LongTensor)
            # Load data to GPU
            labels = labels.to(general_settings['device'])
            questions = questions.to(general_settings['device'])
            output = model(questions, labels)
            loss, _ = output

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update running values
            running_loss += loss.item()
            global_step += 1

            # Evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    # Validation loop
                    for (labels, questions), _ in valid_loader:
                        labels = labels.type(torch.LongTensor)
                        questions = questions.type(torch.LongTensor)
                        # Load data to GPU
                        labels = labels.to(general_settings['device'])
                        questions = questions.to(general_settings['device'])
                        output = model(questions, labels)
                        loss, _ = output
                        valid_running_loss += loss.item()

                # Evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # Resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()

                # Print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                    .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader), average_train_loss, average_valid_loss))

                # Checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/' + 'model.pt', model, best_valid_loss)
                    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)

    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')

# Evaluation Function
def evaluate(model, test_loader):
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for (labels, questions), _ in test_loader:
            labels = labels.type(torch.LongTensor)
            questions = questions.type(torch.LongTensor)
            # Load data to GPU
            labels = labels.to(general_settings['device'])
            questions = questions.to(general_settings['device'])
            output = model(questions, labels)

            _, output = output
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_true.extend(labels.tolist())

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))


# Reproducing same results using torch seed
torch.manual_seed(general_settings['SEED'])
# Cuda algorithms
torch.backends.cudnn.deterministic = True

# Load data and save in temporary JSON files
with open("../../data/LC-QuAD_2_train_balanced.json", "r") as json_file:
    train_data = json.load(json_file)
    get_lists_from_elements(train_data, "train")
with open("../../data/LC-QuAD_2_test_balanced.json", "r") as json_file:
    test_data = json.load(json_file)
    get_lists_from_elements(test_data, "test")

# Define model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Model parameter
MAX_SEQ_LEN = 64
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

# Fields
label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                   fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
fields = {'label': ('label', label_field), 'question': ('question', text_field)}
# Load data
train_set, test_set = TabularDataset.splits(path='./', train='train_temp.json', test='test_temp.json', format='json', fields=fields)
valid_set, train_set = train_set.split(split_ratio=general_settings['valid_data_size'])
# Iterators
train_iter = BucketIterator(train_set, batch_size=general_settings['BATCH_SIZE'], sort_key=lambda x: len(x.question),
                            device=general_settings['device'], train=True, sort=True, sort_within_batch=True)
valid_iter = BucketIterator(valid_set, batch_size=general_settings['BATCH_SIZE'], sort_key=lambda x: len(x.question),
                            device=general_settings['device'], train=True, sort=True, sort_within_batch=True)
test_iter = Iterator(test_set, batch_size=general_settings['BATCH_SIZE'], device=general_settings['device'], train=False, shuffle=False, sort=False)

# Train
model = BERT().to(general_settings['device'])
optimizer = optim.Adam(model.parameters(), lr=5e-4)
train(model=model, optimizer=optimizer, num_epochs=20, train_loader=train_iter, valid_loader=valid_iter)

# Test
best_model = BERT().to(general_settings['device'])
load_checkpoint(general_settings['destination_path'] + '/model.pt', best_model)
evaluate(best_model, test_iter)