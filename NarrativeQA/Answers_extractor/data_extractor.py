import json
import csv

# Get Hard EM predictions
def extract_Hard_EM_predictions(path):
    # Get JSON from file
    with open(path, mode='r', encoding='utf-8') as json_file:
        predictions_json = json.load(json_file)
    predictions = []
    for single_prediction_json in predictions_json.values():
        # Get the fourth prediction, i. e. the one obtained using 80 paragraphs. I didn't notice any difference between
        # the four versions, so I'll get the one that should be more accurate
        predictions.append(single_prediction_json[3])
    return predictions

# Get CommonSense Multi Hop QA predictions
def extract_Multi_Hop_predictions(path):
    predictions = []
    with open(path, mode='r', encoding='utf-8') as file:
        file_lines = file.read().splitlines()
        for line in file_lines:
            predictions.append(line)
    return predictions

# Get test scripts ids (to get summaries), questions and answers
def extract_questions_answers_data(path):
    with open(path, mode='r', encoding='utf-8') as qaps_file:
        qaps = csv.DictReader(qaps_file, delimiter=',')
        documents_ids = []
        questions = []
        first_answers = []
        second_answers = []
        for row in qaps:
            if row["set"] == "test":
                documents_ids.append(row["document_id"])
                questions.append(row["question"])
                first_answers.append(row["answer1"])
                second_answers.append(row["answer2"])

    return documents_ids, questions, first_answers, second_answers
