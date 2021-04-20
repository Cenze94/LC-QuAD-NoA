#!/bin/bash
"""This file is separated from "check_deeppavlov_model_functions.py" to avoid useless libraries loading"""
from deeppavlov import configs, build_model
from typing import List
import json
import os

def load_questions() -> List[str]:
    with open("data/LC-QuAD_2_train_balanced.json", "r") as json_file:
        json_data = json.load(json_file)
    questions = []
    for element in json_data:
        if element['question']:
            questions.append(element['question'])
        else:
            questions.append(element['NNQT_question'])
    return questions


questions = load_questions()
# Delete answers file if exists
if os.path.exists("output/deeppavlov_answers.txt"):
    os.remove("output/deeppavlov_answers.txt")
# Define outputs directory and delete files if exist
os.environ['lc-quad_output_path'] = os.path.abspath(os.getcwd()) + '/output/';
if os.path.exists(os.environ['lc-quad_output_path'] + "queries_candidates.txt"):
    os.remove(os.environ['lc-quad_output_path'] + "queries_candidates.txt")
if os.path.exists(os.environ['lc-quad_output_path'] + "queries_templates.txt"):
    os.remove(os.environ['lc-quad_output_path'] + "queries_templates.txt")
if os.path.exists(os.environ['lc-quad_output_path'] + "answers_indexes.txt"):
    os.remove(os.environ['lc-quad_output_path'] + "answers_indexes.txt")
if os.path.exists(os.environ['lc-quad_output_path'] + "candidate_outputs_lists.txt"):
    os.remove(os.environ['lc-quad_output_path'] + "candidate_outputs_lists.txt")
if os.path.exists(os.environ['lc-quad_output_path'] + "candidate_outputs.txt"):
    os.remove(os.environ['lc-quad_output_path'] + "candidate_outputs.txt")
kbqa_model = build_model(configs.kbqa.kbqa_cq, download=False)
for question in questions:
    try:
        answer = kbqa_model([question])
        with open("output/deeppavlov_answers.txt", "a") as answers_file:
            answers_file.write(str(answer) + "\n")
    except (ValueError) as e:
        with open("output/deeppavlov_answers.txt", "a") as answers_file:
            answers_file.write("['Error:\n" + str(e) + "']")
    # Append a specific string to every support file to associate every lines group to questions
    files = [os.environ['lc-quad_output_path'] + "queries_candidates.txt", os.environ['lc-quad_output_path'] + "queries_templates.txt", \
        os.environ['lc-quad_output_path'] + "answers_indexes.txt", os.environ['lc-quad_output_path'] + "candidate_outputs_lists.txt", \
            os.environ['lc-quad_output_path'] + "candidate_outputs.txt"]
    for file_element in files:
        with open(file_element, "a") as file_handler:
            file_handler.write("-|-\n")