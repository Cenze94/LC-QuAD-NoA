import json
import re
import sqlite3
import random
import ast
from copy import deepcopy
from typing import List, Tuple

# Libraries

import matplotlib.pyplot as plt
import pandas as pd
import torch

# Preliminaries

from torchtext.data import Field, LabelField, TabularDataset, BucketIterator

# Models

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

# Training

import torch.optim as optim

# Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Additional

import numpy as np
from torch.utils.data import TensorDataset, DataLoader

general_settings = {
    # Check whether cuda is available
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "SEED": 2021,
    "BATCH_SIZE": 32,
    "valid_data_size": 0.15,
    "embedding_size": 0,
    "lstm_input_size": 200,
    "lstm_hidden_size": 128,
    "destination_path": "output",
    "embeddings_db_directory": "../../../LC-QuAD-NoA/mini-dataset/mini-dataset_embeddings.db"
}

# Dictionary containing the encodings of all possible templates (encodings values are calculated automatically during the execution)
templates_encoding = {
    "simple question left": {2: []},
    "right-subgraph": {1: [], 2: []},
    "center": {"1.1": [], "1.2": []},
    "rank": {"Rank1": [], "Rank2": []},
    "string matching simple contains word": {1: [], 2: []},
    "statement_property": {"statement_property_1": [], "statement_property_2": [], "Count_1": [], "Count_2": []},
    "simple question right": {1: []},
    "string matching type + relation contains word": {3: [], 4: []},
    "two intentions right subgraph": {"1": []},
    "unknown": {2: [], 3: []},
    "left-subgraph": {5: []}
}

# Support function, query the DB to find if there is the element embedding
def check_element_DB(element_name: str, c: sqlite3.Cursor) -> bool:
    results = c.execute("SELECT * FROM TransE WHERE qid='" + element_name + "'").fetchone()
    return not results is None

# Support function, return the element embedding taken from the DB
def find_element_DB(element_name: str, c: sqlite3.Cursor) -> List[int]:
    query_result = c.execute("SELECT * FROM TransE WHERE qid='" + element_name + "'").fetchone()
    if query_result:
        return list(query_result)
    else:
        return None

# Prepare one-hot encodings of templates
def prepare_encodings():
    templates_number = 0
    for subgraph in templates_encoding.keys():
        templates_number += len(templates_encoding[subgraph])
    one_position = 0
    for subgraph in templates_encoding.keys():
        for template_id in templates_encoding[subgraph].keys():
            templates_encoding[subgraph][template_id] = [0] * templates_number
            templates_encoding[subgraph][template_id][one_position] = 1
            one_position += 1

# Get list of embeddings types in the right order
def get_embeddings_schema_from_query(sparql_query: str) -> List[str]:
    elements = re.findall(r'wd:(\w\d+) |wd:(\w\d+).|wd:(\w\d+)}|wdt:(\w\d+) |p:(\w\d+) |ps:(\w\d+) |pq:(\w\d+) ', sparql_query)
    entities_index = 0
    entities = []
    relations_index = 0
    relations = []
    embeddings_list = []
    for element in elements:
        for possible_value in element:
            if possible_value:
                embeddings_list.append(possible_value)
    return embeddings_list

def preprocess_questions_file(filename: str):
    with open("../../data/LC-QuAD_2_" + filename + "_balanced.json", "r") as json_file:
        json_data = json.load(json_file)
    # Connect to DB
    conn = sqlite3.connect(general_settings['embeddings_db_directory'])
    c = conn.cursor()
    preprocessed_questions = []
    no_embedding_count = 0
    for index, question in enumerate(json_data):
        # If question lacks any embedding, exclude from final data
        no_embedding = False
        preprocessed_question = {
            'uid': question['uid'],
            'sparql_wikidata': question['sparql_wikidata'],
            'subgraph': question['subgraph'],
            'template_id': question['template_id'],
            'template': question['template']
        }
        if question['question']:
            preprocessed_question['question'] = question['question']
        else:
            preprocessed_question['question'] = question['NNQT_question']
        if 'has_answer' in question and question['has_answer'] == False:
            preprocessed_question['answerable'] = 0
        else:
            preprocessed_question['answerable'] = 1
        # Save the right template encoding
        subgraph_value = question['subgraph']
        if not question['subgraph']:
            subgraph_value = "unknown"
        preprocessed_question['template_encoding'] = templates_encoding[subgraph_value][question['template_id']]
        # Find embeddings
        embeddings_list = get_embeddings_schema_from_query(question['sparql_wikidata'])
        for i, element in enumerate(embeddings_list):
            if not no_embedding:
                result = find_element_DB(element, c)
                if result is None:
                    no_embedding_count += 1
                    no_embedding = True
                else:
                    embeddings_list[i] = result[1:]
        if not no_embedding:
            preprocessed_question['embeddings'] = embeddings_list
            preprocessed_questions.append(preprocessed_question)
        if index % 100 == 0:
            print(index)
    c.close()
    print("Questions without at least one embedding: " + str(no_embedding_count) + "/" + str(len(json_data)))
    with open("preprocessed_" + filename + "_balanced.json", "w") as json_file:
        json.dump(preprocessed_questions, json_file, indent=2, ensure_ascii=False)

# Prepare question data, including entities and relation lists
def preprocess_questions():
    prepare_encodings()
    preprocess_questions_file("train")
    preprocess_questions_file("test")

def get_tensors_from_file(filename: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Load labels and embeddings
    with open(filename, "r") as json_file:
        json_data = json.load(json_file)
    # Prepare data for LSTM (embeddings list, template vector and label)
    labels_list = []
    embeddings_list = []
    templates_list = []
    for question in json_data:
        labels_list.append(question['answerable'])
        final_embedding = []
        for embedding in question['embeddings']:
            final_embedding.extend(embedding)
        embeddings_list.append(torch.Tensor(np.array(final_embedding)))
        # Get the right template encoding
        subgraph_value = question['subgraph']
        if not question['subgraph']:
            subgraph_value = "unknown"
        templates_list.append(np.array(templates_encoding[subgraph_value][question['template_id']]))
        # If template encoding length is not saved yet into general_settings, calculate and save it. Do the same for embedding length if the saved value is smaller
        if not "template_encoding_length" in general_settings:
            general_settings['template_encoding_length'] = len(templates_list[-1])
        if general_settings['embedding_size'] < len(final_embedding):
            general_settings['embedding_size'] = len(final_embedding)
    # Add temporarily a tensor with the maximum length of zeros, because the test set might not have any example with the maximum possible embeddings length, and so
    # the input dimension would be wrong. Doing so "pad_sequence" should pad train and test set uniformly. Before returning the embedding list the last element is dropped
    zeros_tensor = torch.tensor((), dtype=torch.float64)
    embeddings_list.append(zeros_tensor.new_zeros(general_settings['embedding_size']))
    embeddings_tensor = pad_sequence(embeddings_list, batch_first=True)[:, None, :]
    return embeddings_tensor[:-1], torch.Tensor(templates_list).float(), torch.FloatTensor(labels_list)

def load_lstm_dataset() -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Get training data from JSON file
    embeddings_tensor, templates_tensor, labels_tensor = get_tensors_from_file("../../data/LC_QuAD_2_train_balanced_with_embeddings.json")
    training_data = TensorDataset(embeddings_tensor, templates_tensor, labels_tensor)
    # Get validation data from training data, splitting randomly a specific percentage of questions
    total_length = len(training_data)
    valid_length = int(total_length * general_settings['valid_data_size'])
    train_length = total_length - valid_length
    train_data, valid_data = torch.utils.data.random_split(training_data, [train_length, valid_length], generator=torch.Generator().manual_seed(general_settings['SEED']))
    # Get test data from JSON file
    embeddings_tensor, templates_tensor, labels_tensor = get_tensors_from_file("../../data/LC_QuAD_2_test_balanced_with_embeddings.json")
    test_data = TensorDataset(embeddings_tensor, templates_tensor, labels_tensor)
    # Print the number of questions for every type
    print("Training set size: " + str(len(train_data)))
    print("Validation set size: " + str(len(valid_data)))
    print("Test set size: " + str(len(test_data)))

    # Data are used in LSTM through DataLoader
    train_loader = DataLoader(train_data.dataset, batch_size = general_settings['BATCH_SIZE'])
    valid_loader = DataLoader(valid_data.dataset, batch_size = general_settings['BATCH_SIZE'])
    test_loader = DataLoader(test_data, batch_size = general_settings['BATCH_SIZE'])

    return train_loader, valid_loader, test_loader

class LSTM(nn.Module):
    def __init__(self, dimension: int = general_settings['lstm_hidden_size']):
        super(LSTM, self).__init__()
        self.dimension = dimension
        # Set LSTM
        self.lstm = nn.LSTM(input_size = general_settings['embedding_size'], hidden_size = dimension, num_layers = 1, batch_first = True, bidirectional = True)
        self.drop = nn.Dropout(p = 0.5)
        # Set additional dense layer
        self.fc_additional = nn.Linear(2 * dimension + general_settings['template_encoding_length'], 2 * dimension + general_settings['template_encoding_length'])
        self.fc_additional_2 = nn.Linear(2 * dimension + general_settings['template_encoding_length'], 2 * dimension + general_settings['template_encoding_length'])
        self.fc_additional_3 = nn.Linear(2 * dimension + general_settings['template_encoding_length'], 2 * dimension + general_settings['template_encoding_length'])
        # Set dense layer for binary classification
        self.fc = nn.Linear(2 * dimension + general_settings['template_encoding_length'], 1)

    def forward(self, embedding, embedding_len, template):
        # Pack embeddings and use the resulting object as LSTM input. Packing is useful and necessary when input elements have different size, this is not the case,
        # unless other templates are added to dataset
        packed_input = pack_padded_sequence(embedding, embedding_len, batch_first = True, enforce_sorted = False)
        packed_output, _ = self.lstm(packed_input)
        # Unpack LSTM result
        output, _ = pad_packed_sequence(packed_output, batch_first = True)

        out_forward = output[range(len(output)), embedding_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        embedding_fea = self.drop(out_reduced)

        # Get dense layer result
        embedding_fea = self.fc_additional(torch.cat((template, embedding_fea), 1))
        embedding_fea = self.fc_additional_2(embedding_fea)
        embedding_fea = self.fc_additional_3(embedding_fea)
        embedding_fea = self.fc(embedding_fea)
        embedding_fea = torch.squeeze(embedding_fea, 1)
        embedding_out = torch.sigmoid(embedding_fea)

        return embedding_out

# Save and load functions
def save_checkpoint(save_path, model, optimizer, valid_loss):
    if save_path == None:
        return
    state_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'valid_loss': valid_loss
    }
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model, optimizer):
    if load_path == None:
        return
    state_dict = torch.load(load_path, map_location = general_settings['device'])
    print(f'Model loaded from <== {load_path}')
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    return state_dict['valid_loss']

def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path == None:
        return
    state_dict = {
        'train_loss_list': train_loss_list,
        'valid_loss_list': valid_loss_list,
        'global_steps_list': global_steps_list
    }
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_metrics(load_path):
    if load_path == None:
        return
    state_dict = torch.load(load_path, map_location = general_settings['device'])
    print(f'Model loaded from <== {load_path}')
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

# Training function
def train(model, optimizer, train_loader, valid_loader,
          criterion = nn.BCELoss(), max_num_epochs = 100, min_num_epochs = 20, num_stop_epochs = 10, eval_every = None,
          file_path = general_settings['destination_path'], best_valid_loss = float("Inf")):
    # Initialize running values
    if not eval_every:
        # This value indicates the number of examples to execute for training before validation and possible save. Currently it's done at the end of every epoch
        eval_every = len(train_loader)
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # Training loop
    model.train()
    epoch = 0
    stop_epoch = 0
    while epoch < max_num_epochs and stop_epoch < num_stop_epochs or epoch < min_num_epochs:
        updated = False
        for embeddings, templates, labels in train_loader:
            # Prepare input sizes tensor for LSTM
            embeddings_sizes = list(embeddings.shape)
            embeddings_sizes_list = []
            for i in range(embeddings_sizes[0]):
                embeddings_sizes_list.append(embeddings_sizes[1])
            embeddings_len = torch.LongTensor(np.array(embeddings_sizes_list))
            # Load data to GPU
            labels = labels.to(general_settings['device'])
            embeddings = embeddings.to(general_settings['device'])
            templates = templates.to(general_settings['device'])
            # Execute model
            output = model(embeddings, embeddings_len, templates)

            loss = criterion(output, labels)
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
                    for embeddings, templates, labels in train_loader:
                        # Prepare input sizes tensor for LSTM
                        embeddings_sizes = list(embeddings.shape)
                        embeddings_sizes_list = []
                        for i in range(embeddings_sizes[0]):
                            embeddings_sizes_list.append(embeddings_sizes[1])
                        embeddings_len = torch.LongTensor(np.array(embeddings_sizes_list))
                        # Load data to GPU
                        labels = labels.to(general_settings['device'])
                        embeddings = embeddings.to(general_settings['device'])
                        templates = templates.to(general_settings['device'])
                        # Execute model
                        output = model(embeddings, embeddings_len, templates)

                        loss = criterion(output, labels)
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
                print('Epoch [{}/{}], Stop Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                    .format(epoch+1, max_num_epochs, stop_epoch, num_stop_epochs, global_step, max_num_epochs*len(train_loader), average_train_loss, average_valid_loss))

                # Checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + "/model.pt", model, optimizer, best_valid_loss)
                    save_metrics(file_path + "/metrics.pt", train_loss_list, valid_loss_list, global_steps_list)
                    updated = True
        if updated:
            stop_epoch = 0
        else:
            stop_epoch += 1
        epoch += 1

    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')

# Show training and validation loss plots
def show_plots():
    train_loss_list, valid_loss_list, global_steps_list = load_metrics(general_settings['destination_path'] + '/metrics.pt')
    plt.plot(global_steps_list, train_loss_list, label='Train')
    plt.plot(global_steps_list, valid_loss_list, label='Valid')
    plt.xlabel('Global Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Evaluation function with test set
def evaluate(model, test_loader, threshold = 0.5):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for embeddings, templates, labels in test_loader:
            # Prepare input sizes tensor for LSTM
            embeddings_sizes = list(embeddings.shape)
            embeddings_sizes_list = []
            for i in range(embeddings_sizes[0]):
                embeddings_sizes_list.append(embeddings_sizes[1])
            embeddings_len = torch.LongTensor(np.array(embeddings_sizes_list))
            # Load data to GPU
            labels = labels.to(general_settings['device'])
            embeddings = embeddings.to(general_settings['device'])
            templates = templates.to(general_settings['device'])
            # Execute model
            output = model(embeddings, embeddings_len, templates)

            # If output value exceeds the threshold, the model considers the question answerable
            output = (output > threshold).int()
            # Save data for metrics print
            y_pred.extend(output.tolist())
            y_true.extend(labels.tolist())

    # Metrics print
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))

    # Write predictions to file
    with open("model_predictions.txt", "w") as answers_file:
        answers_file.write(str(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels = [1, 0])
    ax = plt.subplot()
    sns.heatmap(cm, annot = True, ax = ax, cmap = 'Blues', fmt = 'd')
    ax.set_title('Confusion matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
    ax.yaxis.set_ticklabels(['FAKE', 'REAL'])

# Code for reproducibility
"""# Reproducing same results using torch seed
torch.manual_seed(general_settings['SEED'])
# Cuda algorithms
torch.backends.cudnn.deterministic = True"""

###### Prepare dataset ######

#preprocess_questions()

###### Run classification #######

train_loader, valid_loader, test_loader = load_lstm_dataset()
model = LSTM().to(general_settings['device'])
optimizer = optim.Adam(model.parameters(), lr = 0.00075)
train(model = model, optimizer = optimizer, train_loader = train_loader, valid_loader = valid_loader, max_num_epochs = 300, num_stop_epochs = 10)
#show_plots()

best_model = LSTM().to(general_settings['device'])
load_checkpoint(general_settings['destination_path'] + '/model.pt', best_model, optimizer)
evaluate(best_model, test_loader)

# Add False predicted labels for questions excluded because of the lack of one or more embeddings, following the right order
with open("model_predictions.txt", "r") as answers_file:
    predicted_answers =  ast.literal_eval(answers_file.readline())
with open("../../data/LC-QuAD_2_test_balanced.json", "r") as json_file:
    original_dataset_questions = json.load(json_file)
with open("preprocessed_test_balanced.json", "r") as json_file:
    preprocessed_dataset_questions = json.load(json_file)

# Check IDs to find excluded questions
for index, element in enumerate(original_dataset_questions):
    found = False
    i = 0
    while not found and i < len(preprocessed_dataset_questions):
        if element['uid'] == preprocessed_dataset_questions[i]['uid']:
            found = True
        i += 1
    if not found:
        predicted_answers.insert(index, 0)
with open("model_predictions.txt", "w") as answers_file:
    answers_file.write(str(predicted_answers))