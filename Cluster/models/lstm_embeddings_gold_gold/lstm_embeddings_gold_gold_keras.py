import json
import re
import sqlite3
import random
from copy import deepcopy
from typing import List, Tuple

# Libraries

import matplotlib.pyplot as plt
import pandas as pd

# Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Additional

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

general_settings = {
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

def get_tensors_from_file(filename: str) -> Tuple[np.array, np.array, np.array]:
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
        embeddings_list.append(np.array(final_embedding))
        templates_list.append(np.array(question['template_encoding']))
        # If template encoding length is not saved yet into general_settings, calculate and save it. Do the same for embedding length if the saved value is smaller
        if not "template_encoding_length" in general_settings:
            general_settings['template_encoding_length'] = len(question['template_encoding'])
        if general_settings['embedding_size'] < len(final_embedding):
            general_settings['embedding_size'] = len(final_embedding)
    return np.expand_dims(sequence.pad_sequences(np.array(embeddings_list), maxlen=general_settings['embedding_size'], dtype='float64'), axis=1), np.array(templates_list), np.array(labels_list)

def load_lstm_dataset() -> Tuple[Tuple[np.array, np.array, np.array], Tuple[np.array, np.array, np.array], Tuple[np.array, np.array, np.array]]:
    # Get training data from JSON file
    embeddings_tensor, templates_tensor, labels_tensor = get_tensors_from_file("preprocessed_train_balanced.json")
    # Get validation data from training data, splitting randomly a specific percentage of questions
    embeddings_tensor_train, embeddings_tensor_valid, templates_tensor_train, templates_tensor_valid, labels_tensor_train, labels_tensor_valid = train_test_split(\
        embeddings_tensor, templates_tensor, labels_tensor, test_size=general_settings['valid_data_size'], random_state=general_settings['SEED'])
    train_data = (embeddings_tensor_train, templates_tensor_train, labels_tensor_train)
    valid_data = (embeddings_tensor_valid, templates_tensor_valid, labels_tensor_valid)
    # Get test data from JSON file
    embeddings_tensor, templates_tensor, labels_tensor = get_tensors_from_file("preprocessed_test_balanced.json")
    test_data = (embeddings_tensor, templates_tensor, labels_tensor)
    # Print the number of questions for every type
    print("Training set size: " + str(len(train_data[2])))
    print("Validation set size: " + str(len(valid_data[2])))
    print("Test set size: " + str(len(test_data[2])))

    return train_data, valid_data, test_data

# Reproducing same results using seed
np.random.seed(general_settings['SEED'])

###### Prepare dataset ######

#preprocess_questions()

###### Run classification #######

train_data, valid_data, test_data = load_lstm_dataset()
model = Sequential()
model.add(Dropout(0.5))
model.add(LSTM(general_settings['lstm_hidden_size']))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_data[0], train_data[2], epochs=20, batch_size=general_settings['BATCH_SIZE'], validation_data=(valid_data[0], valid_data[2]))
print(model.summary())
# Final evaluation of the model
scores = model.evaluate(test_data[0], test_data[2], verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))