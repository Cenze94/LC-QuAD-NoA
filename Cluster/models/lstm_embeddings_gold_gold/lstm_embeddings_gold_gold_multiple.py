import json
import re
import sqlite3
import random
from copy import deepcopy
from typing import List, Tuple, Dict, Any

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
    "lstm_hidden_size": 128,
    "destination_path": "output",
    "embeddings_db_directory": "../../../LC-QuAD-NoA/mini-dataset/mini-dataset_embeddings.db"
}

# Dictionary containing data of all possible templates
templates = {
    "simple question left": {2: {}},
    "right-subgraph": {1: {}, 2: {}},
    "center": {"1.1": {}, "1.2": {}},
    "rank": {"Rank1": {}, "Rank2": {}},
    "string matching simple contains word": {1: {}, 2: {}},
    "statement_property": {"statement_property_1": {}, "statement_property_2": {}, "Count_1": {}, "Count_2": {}},
    "simple question right": {1: {}},
    "string matching type + relation contains word": {3: {}},
    "two intentions right subgraph": {"1": {}},
    "unknown": {2: {}, 3: {}},
    "left-subgraph": {5: {}}
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

# Get list of embeddings types in the right order, entities list and relations list
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
    with open("preprocessed_multiple_" + filename + "_balanced.json", "w") as json_file:
        json.dump(preprocessed_questions, json_file, indent=2, ensure_ascii=False)

# Prepare question data, including entities and relation lists
def preprocess_questions():
    preprocess_questions_file("train")
    preprocess_questions_file("test")

def get_subgraph_and_template_id_from_question(question: Dict[str, Any]) -> Tuple[str, str]:
    if question['subgraph']:
        # In test set there aren't "string matching type + relation contains word" of type 4 (with letter check)
        if question['subgraph'] == "string matching type + relation contains word":
            return question['subgraph'], 3
        return question['subgraph'], question['template_id']
    else:
        return "unknown", question['template_id']

def get_tensors_from_file(filename: str, file_type: str):
    # Load labels and embeddings
    with open(filename, "r") as json_file:
        json_data = json.load(json_file)
    # Prepare data for LSTM (embeddings list, template vector and label)
    for template in templates.keys():
        for template_id in templates[template].keys():
            templates[template][template_id][file_type]['labels_list'] = []
            templates[template][template_id][file_type]['embeddings_list'] = []
    for question in json_data:
        subgraph, template_id = get_subgraph_and_template_id_from_question(question)
        templates[subgraph][template_id][file_type]['labels_list'].append(question['answerable'])
        final_embedding = []
        for embedding in question['embeddings']:
            final_embedding.extend(embedding)
        templates[subgraph][template_id][file_type]['embeddings_list'].append(torch.Tensor(np.array(final_embedding)))
        # If template encoding length is not saved yet into general_settings, calculate and save it. Do the same for embedding length if the saved value is smaller
        if not 'embedding_size' in templates[subgraph][template_id] or templates[subgraph][template_id]['embedding_size'] < len(final_embedding):
            templates[subgraph][template_id]['embedding_size'] = len(final_embedding)
    for template in templates.keys():
        for template_id in templates[template].keys():
            if not templates[template][template_id][file_type]['embeddings_list']:
                print(template, str(template_id))
            templates[template][template_id][file_type]['embeddings_list'] = pad_sequence(templates[template][template_id][file_type]['embeddings_list'], batch_first=True)[:, None, :]
            templates[template][template_id][file_type]['labels_list'] = torch.FloatTensor(templates[template][template_id][file_type]['labels_list'])

def load_lstm_dataset():
    for template in templates.keys():
        for template_id in templates[template].keys():
            templates[template][template_id]["train"] = {}
            templates[template][template_id]["test"] = {}
    # Get training data from JSON file
    get_tensors_from_file("preprocessed_multiple_train_balanced.json", "train")
    # Get test data from JSON file
    get_tensors_from_file("preprocessed_multiple_test_balanced.json", "test")
    for template in templates.keys():
        for template_id in templates[template].keys():
            training_data = TensorDataset(templates[template][template_id]["train"]['embeddings_list'], templates[template][template_id]["train"]['labels_list'])
            # Get validation data from training data, splitting randomly a specific percentage of questions
            total_length = len(training_data)
            valid_length = int(total_length * general_settings['valid_data_size'])
            train_length = total_length - valid_length
            train_data, valid_data = torch.utils.data.random_split(training_data, [train_length, valid_length], generator=torch.Generator().manual_seed(general_settings['SEED']))
            test_data = TensorDataset(templates[template][template_id]["test"]['embeddings_list'], templates[template][template_id]["test"]['labels_list'])
            # Print the number of questions for every type
            print(template + " " + str(template_id))
            print("Training set size: " + str(len(train_data)))
            print("Validation set size: " + str(len(valid_data)))
            print("Test set size: " + str(len(test_data)))

            # Data are used in LSTM through DataLoader
            templates[template][template_id]['train_loader'] = DataLoader(train_data.dataset, batch_size = general_settings['BATCH_SIZE'])
            templates[template][template_id]['valid_loader'] = DataLoader(valid_data.dataset, batch_size = general_settings['BATCH_SIZE'])
            templates[template][template_id]['test_loader'] = DataLoader(test_data, batch_size = general_settings['BATCH_SIZE'])

class LSTM(nn.Module):
    def __init__(self, template: str, template_id: Any, dimension: int = general_settings['lstm_hidden_size']):
        super(LSTM, self).__init__()
        self.dimension = dimension
        self.template = template
        self.template_id = template_id
        # Set LSTM
        self.lstm = nn.LSTM(input_size = templates[template][template_id]['embedding_size'], hidden_size = dimension, num_layers = 1, batch_first = True, bidirectional = True)
        self.drop = nn.Dropout(p = 0.5)
        # Set dense layer for binary classification
        self.fc = nn.Linear(2 * dimension, 1)

    def forward(self, embedding, embedding_len):
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
    #print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model, optimizer):
    if load_path == None:
        return
    state_dict = torch.load(load_path, map_location = general_settings['device'])
    #print(f'Model loaded from <== {load_path}')
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
    #print(f'Model saved to ==> {save_path}')

def load_metrics(load_path):
    if load_path == None:
        return
    state_dict = torch.load(load_path, map_location = general_settings['device'])
    #print(f'Model loaded from <== {load_path}')
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

# Training function
def train(model, optimizer, train_loader, valid_loader,
          criterion = nn.BCELoss(), num_epochs = 5, eval_every = None,
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
    for epoch in range(num_epochs):
        for embeddings, labels in train_loader:
            # Prepare input sizes tensor for LSTM
            embeddings_sizes = list(embeddings.shape)
            embeddings_sizes_list = []
            for i in range(embeddings_sizes[0]):
                embeddings_sizes_list.append(embeddings_sizes[1])
            embeddings_len = torch.LongTensor(np.array(embeddings_sizes_list))
            # Load data to GPU
            labels = labels.to(general_settings['device'])
            embeddings = embeddings.to(general_settings['device'])
            # Execute model
            output = model(embeddings, embeddings_len)

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
                    for embeddings, labels in train_loader:
                        # Prepare input sizes tensor for LSTM
                        embeddings_sizes = list(embeddings.shape)
                        embeddings_sizes_list = []
                        for i in range(embeddings_sizes[0]):
                            embeddings_sizes_list.append(embeddings_sizes[1])
                        embeddings_len = torch.LongTensor(np.array(embeddings_sizes_list))
                        # Load data to GPU
                        labels = labels.to(general_settings['device'])
                        embeddings = embeddings.to(general_settings['device'])
                        # Execute model
                        output = model(embeddings, embeddings_len)

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
                """print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                    .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader), average_train_loss, average_valid_loss))"""

                # Checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + "/model" + template + '_' + str(template_id) + ".pt", model, optimizer, best_valid_loss)
                    save_metrics(file_path + "/metrics" + template + '_' + str(template_id) + ".pt", train_loss_list, valid_loss_list, global_steps_list)

    save_metrics(file_path + '/metrics' + template + '_' + str(template_id) + '.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')

# Show training and validation loss plots
def show_plots(template: str, template_id: Any):
    train_loss_list, valid_loss_list, global_steps_list = load_metrics(general_settings['destination_path'] + '/metrics' + template + '_' + str(template_id) + '.pt')
    plt.plot(global_steps_list, train_loss_list, label='Train')
    plt.plot(global_steps_list, valid_loss_list, label='Valid')
    plt.xlabel('Global Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Evaluation function with test set
def evaluate(model, test_loader, version = 'title', threshold = 0.5) -> Tuple[List[int], List[int]]:
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for embeddings, labels in test_loader:
            # Prepare input sizes tensor for LSTM
            embeddings_sizes = list(embeddings.shape)
            embeddings_sizes_list = []
            for i in range(embeddings_sizes[0]):
                embeddings_sizes_list.append(embeddings_sizes[1])
            embeddings_len = torch.LongTensor(np.array(embeddings_sizes_list))
            # Load data to GPU
            labels = labels.to(general_settings['device'])
            embeddings = embeddings.to(general_settings['device'])
            # Execute model
            output = model(embeddings, embeddings_len)

            # If output value exceeds the threshold, the model considers the question answerable
            output = (output > threshold).int()
            # Save data for metrics print
            y_pred.extend(output.tolist())
            y_true.extend(labels.tolist())
    # Metrics print
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))

    """cm = confusion_matrix(y_true, y_pred, labels = [1, 0])
    ax = plt.subplot()
    sns.heatmap(cm, annot = True, ax = ax, cmap = 'Blues', fmt = 'd')
    ax.set_title('Confusion matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
    ax.yaxis.set_ticklabels(['FAKE', 'REAL'])"""

    return y_pred, y_true

# Reproducing same results using torch seed
torch.manual_seed(general_settings['SEED'])
# Cuda algorithms
torch.backends.cudnn.deterministic = True

###### Prepare dataset ######

#preprocess_questions()

###### Run classification #######

load_lstm_dataset()
y_pred_final = []
y_true_final = []
for template in templates.keys():
    for template_id in templates[template].keys():
        print(template + " " + str(template_id))
        model = LSTM(template, template_id).to(general_settings['device'])
        optimizer = optim.Adam(model.parameters(), lr = 0.001)
        train(model = model, optimizer = optimizer, train_loader = templates[template][template_id]['train_loader'],
        valid_loader = templates[template][template_id]['valid_loader'], num_epochs = 25)
        #show_plots(template, template_id)

        best_model = LSTM(template, template_id).to(general_settings['device'])
        load_checkpoint(general_settings['destination_path'] + '/model' + template + '_' + str(template_id) + '.pt', best_model, optimizer)
        y_pred, y_true = evaluate(best_model, templates[template][template_id]['test_loader'])
        y_pred_final.extend(y_pred)
        y_true_final.extend(y_true)
print('Final Classification Report:')
print(classification_report(y_true_final, y_pred_final, labels=[1,0], digits=4))