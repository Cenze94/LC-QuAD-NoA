import main
import questions_generator
import simple_question_left_questions_generator as simple_question_left
import right_subgraph_questions_generator as right_subgraph
import center_questions_generator as center
import rank_questions_generator as rank
import string_matching_simple_contains_word_questions_generator as string_matching_simple_contains_word
import statement_property_questions_generator as statement_property
import count_questions_generator as count
import simple_question_right_questions_generator as simple_question_right
import string_matching_type_relation_contains_word_questions_generator as string_matching_type_relation_contains_word
import two_intentions_right_subgraph_questions_generator as two_intentions_right_subgraph
import unknown_questions_generator as unknown

from SPARQLWrapper.SPARQLExceptions import QueryBadFormed
from urllib.error import URLError
from typing import List, Dict, Any, Tuple
import argparse
import subprocess
import json
import re
import os
import random

# Check if input element is an entity, and if so return the corresponding English label
def convert_entity_qid_to_label(element: str) -> str:
    if len(re.findall(r"^Q\d+$", element)) > 0:
        # Element is an entity, return its label
        return questions_generator.get_entity_name_from_wikidata_id(element)
    else:
        # Element is a number, date or another type of string, so return without modifications
        return element

# Create a new JSON file without questions that don't have an answer
def filter_questions(filename: str):
    with open("dataset/" + filename + ".json", "r") as json_file:
        json_data = json.load(json_file)
    if os.path.exists("dataset/" + filename + "_filtered.json"):
        with open("dataset/" + filename + "_filtered.json", "r") as json_file:
            filtered_json_data = json.load(json_file)
    else:
        filtered_json_data = {"num_filtered_questions": -1, "questions": []}
    old_filtered_json_data_length = filtered_json_data['num_filtered_questions']
    for index, element in enumerate(json_data):
        # Skip already saved data
        if index > old_filtered_json_data_length:
            if index % 100 == 0:
                print(index)
            # There are some questions considered malformed, which have to be skipped and so an empty list is returned. The same is done with
            # queries that cause an Internal Server Error
            try:
                results = questions_generator.get_sparql_query_results(element['sparql_wikidata'])
            except QueryBadFormed:
                results = []
            except URLError:
                results = questions_generator.get_sparql_query_results(element['sparql_wikidata'])
            if len(results) > 0:
                if 'boolean' in results:
                    filtered_json_data['questions'].append(element)
                elif len(results['results']['bindings']) > 0 and ((element['template_id'] != "Count_1" and element['template_id'] != "Count_2") or \
                    results['results']['bindings'][0]['value']['value'] != "0"):
                    # Check if there is at least one English answer in cases different from count, otherwise check if the obtained number is different from 0
                    answers = results['results']['bindings']
                    # Check all answers, for a maximum of 50 elements, until it is found one with an English label, since getting labels is a slow process and there
                    # could be questions with a lot of possible answers (I found one with more than 14,000 answers)
                    answer = ""
                    i = 0
                    while i in range(min(50, len(answers))) and not answer:
                        answer_iter = iter(results[i].values())
                        answer = convert_entity_qid_to_label(next(answer_iter)['value'].split("/")[-1])
                        # Check if there is a second answer and has an English label, if so add to answers list (DeepPavlov always returns only a single answer). If an answer
                        # has already been found this check is useless
                        if not answer and (not filtered_question['subgraph'] or filtered_question['subgraph'] == "two intentions right subgraph"):
                            answer = convert_entity_qid_to_label(next(answer_iter)['value'].split("/")[-1])
                        i += 1
                    if answer:
                        filtered_json_data['questions'].append(element)
            filtered_json_data['num_filtered_questions'] += 1
            # Save temporary results
            if index % 5000 == 0 and index > 0:
                with open("dataset/" + filename + "_filtered.json", "w") as json_file:
                    json.dump(filtered_json_data, json_file, indent=2, ensure_ascii=False)
    # Save only questions list
    with open("dataset/" + filename + "_filtered.json", "w") as json_file:
        json.dump(filtered_json_data["questions"], json_file, indent=2, ensure_ascii=False)

# Generated a dataset with already generated questions and the corresponding original questions. It's considered balanced since it's taken for granted that there aren't
# (a lot of) multiple generated questions obtained from the same original element
def create_balanced_dataset(set_name: str):
    # Load filtered questions
    with open("dataset/" + set_name + "_filtered.json", "r") as json_file:
        filtered_questions = json.load(json_file)
    # Load generated questions
    with open("dataset/" + set_name + "_generated.json", "r") as json_file:
        generated_questions = json.load(json_file)
    # Get the filtered questions used to create the generated questions
    original_questions = []
    for index, generated_question in enumerate(generated_questions):
        if index % 50 == 0 and index != 0:
            print(index)
        for filtered_question in filtered_questions:
            # For the check we could use respectively only "old_sparql_wikidata" and "sparql_wikidata", but that choice should be longer to execute than the actual one.
            # This is the reason of the actual checks order, since usually "template_id" and "template_index" values are very short and variable. The last check avoids
            # duplicates if there are questions generated from the same element
            if generated_question['template_id'] == filtered_question['template_id'] and generated_question['template_index'] == filtered_question['template_index'] and \
                generated_question['subgraph'] == filtered_question['subgraph'] and filtered_question not in original_questions:
                original_questions.append(filtered_question)
                break
    # Save questions in random order
    dataset_questions = original_questions + generated_questions
    random.shuffle(dataset_questions)
    with open("dataset/" + set_name + "_balanced.json", "w") as json_file:
        json.dump(dataset_questions, json_file, indent=2, ensure_ascii=False)

# Check original questions to find ones without answer (for example because of changes in Wikidata during dataset creation). Print their uid and the uid of the corresponding
# generated questions, in order to correct them manually
def check_original_questions(set_name: str):
    # Load filtered questions
    with open("dataset/" + set_name + "_filtered.json", "r") as json_file:
        filtered_questions = json.load(json_file)
    # Load generated questions
    with open("dataset/" + set_name + "_generated.json", "r") as json_file:
        generated_questions = json.load(json_file)
    # Get the filtered questions used to create the generated questions
    original_questions = []
    for generated_question in generated_questions:
        for filtered_question in filtered_questions:
            # For the check we could use respectively only "old_sparql_wikidata" and "sparql_wikidata", but that choice should be longer to execute than the actual one.
            # This is the reason of the actual checks order, since usually "template_id" and "template_index" values are very short and variable. The last check avoids
            # duplicates if there are questions generated from the same element
            if generated_question['template_id'] == filtered_question['template_id'] and generated_question['template_index'] == filtered_question['template_index'] and \
                generated_question['subgraph'] == filtered_question['subgraph']:
                # There are some questions considered malformed, which have to be skipped and so an empty list is returned. The same is done with
                # queries that cause an Internal Server Error
                try:
                    results = questions_generator.get_sparql_query_results(filtered_question['sparql_wikidata'])
                except QueryBadFormed:
                    results = []
                except URLError:
                    results = questions_generator.get_sparql_query_results(filtered_question['sparql_wikidata'])
                if len(results) == 0:
                    # There aren't answers
                    print("(" + str(filtered_question['uid']) + ", " + str(generated_question['uid']) + ")")
                elif (filtered_question['template_id'] == "Count_1" or filtered_question['template_id'] == "Count_2") and results['results']['bindings'][0]['value']['value'] == "0":
                    # The question is a count and the answer is 0
                    print("(" + str(filtered_question['uid']) + ", " + str(generated_question['uid']) + ")")
                else:
                    # Check if there is at least one answer with an English label
                    answers = results['results']['bindings']
                    # Check all answers, for a maximum of 50 elements, until it is found one with an English label, since getting labels is a slow process and there
                    # could be questions with a lot of possible answers (I found one with more than 14,000 answers)
                    answer = ""
                    i = 0
                    while i in range(min(50, len(answers))) and not answer:
                        answer_iter = iter(results[i].values())
                        answer = convert_entity_qid_to_label(next(answer_iter)['value'].split("/")[-1])
                        # Check if there is a second answer and has an English label, if so add to answers list (DeepPavlov always returns only a single answer). If an answer
                        # has already been found this check is useless
                        if not answer and (not filtered_question['subgraph'] or filtered_question['subgraph'] == "two intentions right subgraph"):
                            answer = convert_entity_qid_to_label(next(answer_iter)['value'].split("/")[-1])
                        i += 1
                    if not answer:
                        # All answers don't have an English label
                        print("(" + str(filtered_question['uid']) + ", " + str(generated_question['uid']) + ")")
                break

# Check template type, since this project types and LC-QuAD 2.0 ones are different
def check_template_type(template_type: str, question: Dict[str, Any]) -> bool:
    if (template_type == "simple_question_left" and question['subgraph'] == "simple question left") or \
        (template_type == "right_subgraph" and question['subgraph'] == "right-subgraph" and question['template_id'] == 1) or \
        (template_type == "right_subgraph_2" and ((question['subgraph'] == "right-subgraph" and question['template_id'] == 2) or question['subgraph'] == "left-subgraph")) or \
        (template_type == "center" and question['subgraph'] == "center" and question['template_id'] == "1.2") or \
        (template_type == "center_2" and question['subgraph'] == "center" and question['template_id'] == "1.1") or \
        (template_type == "rank" and question['subgraph'] == "rank" and question['template_id'] == "Rank1") or \
        (template_type == "rank_2" and question['subgraph'] == "rank" and question['template_id'] == "Rank2") or \
        (template_type == "string_matching_simple_contains_word" and question['subgraph'] == "string matching simple contains word") or \
        (template_type == "statement_property" and question['subgraph'] == "statement_property" and question['template_id'] == "statement_property_2") or \
        (template_type == "statement_property_2" and question['subgraph'] == "statement_property" and question['template_id'] == "statement_property_1") or \
        (template_type == "count" and question['subgraph'] == "statement_property" and question['template_id'] == "Count_1") or \
        (template_type == "count_2" and question['subgraph'] == "statement_property" and question['template_id'] == "Count_2") or \
        (template_type == "simple_question_right" and question['subgraph'] == "simple question right") or \
        (template_type == "string_matching_type_relation_contains_word" and question['subgraph'] == "string matching type + relation contains word") or \
        (template_type == "two_intentions_right_subgraph" and question['subgraph'] == "two intentions right subgraph") or \
        (template_type == "unknown" and not question['subgraph'] and question['template_id'] == 3) or \
        (template_type == "unknown_2" and not question['subgraph'] and question['template_id'] == 2):
        return True
    return False

# Get a proportional number of questions to generate, based on the respective proportion in the original dataset, and the set of possible candidates
# (i. e. the questions of the given template type)
def get_template_candidates_and_number(template_type: str, examples_num: int) -> Tuple[List[Dict[str, Any]], int, List[Dict[str, Any]], int]:
    # Get total questions numbers
    with open("dataset/train_filtered.json", "r") as json_file:
        json_data_train = json.load(json_file)
    total_num_questions_train = len(json_data_train)
    # Subtract boolean questions number from total, since they are not generated
    for element in json_data_train:
        if element['subgraph'] and 'boolean' in element['subgraph']:
            total_num_questions_train -= 1
    with open("dataset/test_filtered.json", "r") as json_file:
        json_data_test = json.load(json_file)
    total_num_questions_test = len(json_data_test)
    # Subtract boolean questions number from total, since they are not generated
    for element in json_data_test:
        if element['subgraph'] and 'boolean' in element['subgraph']:
            total_num_questions_test -= 1

    # Get total examples numbers for train and test
    examples_num_train = total_num_questions_train / (total_num_questions_train + total_num_questions_test) * examples_num
    examples_num_test = examples_num - examples_num_train

    # Get possible candidates
    train_candidates = []
    for element in json_data_train:
        if check_template_type(template_type, element):
            train_candidates.append(element)
    test_candidates = []
    for element in json_data_test:
        if check_template_type(template_type, element):
            test_candidates.append(element)
    
    # Get number of examples to generate
    original_proportion_train = len(train_candidates) / total_num_questions_train
    num_questions_train = original_proportion_train * examples_num_train
    original_proportion_test = len(test_candidates) / total_num_questions_test
    num_questions_test = original_proportion_test * examples_num_test

    return train_candidates, num_questions_train, test_candidates, num_questions_test

# Support function to get the name of the function to call
def get_function_name(template_type: str, operation_type: str) -> str:
    # There could be variations of the same template that are identified with a number, if they are present get the module name removing that number
    template_variation = re.findall(r'(_\d+)', template_type)
    if len(template_variation) > 0:
        module_name = template_type.replace(template_variation[-1], "")
    else:
        module_name = template_type
    # Get function call from input arguments, assuming that arguments and template remain the same
    return module_name + "." + template_type + "_" + operation_type + "_generation"

# Define expected terminal arguments
parser = argparse.ArgumentParser()
parser.add_argument("--n", help="total number of generated examples", type=int, default=2000)
parser.add_argument("--type", help="type of template to generate", type=str, default="simple_question_left")
args = parser.parse_args()

if args.type == "db_filtering":
    filter_questions("train")
    filter_questions("test")
elif args.type == "check_original_questions":
    print("Checking train set:")
    check_original_questions("train")
    print("Checking test set:")
    check_original_questions("test")
elif args.type == "check_calculations":
    templates = main.templateTypes.keys()
    train_total_num = 0
    test_total_num = 0
    for template in templates:
        _, train_num, _, test_num = get_template_candidates_and_number(template, args.n)
        print(template + " train: " + str(int(train_num)))
        print(template + " test: " + str(int(test_num)))
        train_total_num += train_num
        test_total_num += test_num
    print("total train: " + str(int(train_total_num)))
    print("total test: " + str(int(test_total_num)))
    print("total: " + str(int(train_total_num + test_total_num)))
elif args.type == "create_balanced_dataset":
    create_balanced_dataset("train")
    create_balanced_dataset("test")
elif args.type not in main.templateTypes.keys():
    print("Type not recognized")
else:
    # Get all operation types
    operation_types_list = ["entity", "relation", "entity_2", "relation_2", "entity_3", "relation_3"]
    # Get saved temp data if exists
    if os.path.exists("dataset/temp_state.json"):
        with open("dataset/temp_state.json", "r") as json_file:
            temp_state = json.load(json_file)
        # Temp data is considered valid only if type corresponds
        if temp_state['type'] == args.type:
            # Number of total generated examples must be the same
            if temp_state['n'] != args.n:
                raise SystemExit('Different number of examples saved in temp_state.json')
            # Remove already executed operations from list
            operation_type_index = operation_types_list.index(temp_state['operation_type'])
            for i in range(operation_type_index + 1):
                del operation_types_list[0]
        else:
            temp_state = {"train_iter_position": 0, "test_iter_position": 0}
    else:
        temp_state = {"train_iter_position": 0, "test_iter_position": 0}

    # Get questions candidates and number of examples to generate for train and test
    candidates_train, num_questions_train, candidates_test, num_questions_test = get_template_candidates_and_number(args.type, args.n)
    # Get questions number per operation type, if the obtained value is 0 then change to 1
    num_questions_op_train = int(num_questions_train / 6)
    num_questions_op_test = int(num_questions_test / 6)
    if num_questions_op_train == 0:
        num_questions_op_train = 1
    if num_questions_op_test == 0:
        num_questions_op_test = 1

    # Get generated questions, iterating all elements until a sufficient number of questions is obtained
    current_uid = main.get_current_max_uid()
    candidates_train_iter = iter(candidates_train)
    candidates_train_iter_position = temp_state["train_iter_position"]
    for i in range(temp_state["train_iter_position"]):
        next(candidates_train_iter)
    candidates_test_iter = iter(candidates_test)
    candidates_test_iter_position = temp_state["test_iter_position"]
    for i in range(temp_state["test_iter_position"]):
        next(candidates_test_iter)
    # Load already generated questions
    if os.path.exists("dataset/train_generated.json"):
        with open("dataset/train_generated.json", "r") as json_file:
            generated_questions_train = json.load(json_file)
    else:
        generated_questions_train = []
    if os.path.exists("dataset/test_generated.json"):
        with open("dataset/test_generated.json", "r") as json_file:
            generated_questions_test = json.load(json_file)
    else:
        generated_questions_test = []
    for operation_type in operation_types_list:
        print("Generating " + str(num_questions_op_train) + " questions for train set with " + operation_type + ":")
        for cont in range(num_questions_op_train):
            found = False
            while not found:
                # Return the first element iterator as default value
                original_question = next(candidates_train_iter, next(iter(candidates_train)))
                candidates_train_iter_position += 1
                # If iterator started from beginning, reset index
                if candidates_train_iter_position == len(candidates_train):
                    candidates_train_iter_position = 0
                exec("generated_question = " + get_function_name(args.type, operation_type) + "(current_uid, original_question, generated_questions_train)")
                # Check if the generated question is of the desired difficulty, if not so try with the next example
                if '_2' in operation_type and generated_question['operation_difficulty'] == 2:
                    found = True
                elif '_3' in operation_type and generated_question['operation_difficulty'] == 3:
                    found = True
                elif '_2' not in operation_type and '_3' not in operation_type:
                    found = True
            print("Generated!")
            generated_questions_train.append(generated_question)
            current_uid += 1

        print("Generating " + str(num_questions_op_test) + " questions for test set with " + operation_type + ":")
        for cont in range(num_questions_op_test):
            found = False
            while not found:
                # Return the first element iterator as default value
                original_question = next(candidates_test_iter, next(iter(candidates_test)))
                candidates_test_iter_position += 1
                # If iterator started from beginning, reset index
                if candidates_test_iter_position == len(candidates_test):
                    candidates_test_iter_position = 0
                exec("generated_question = " + get_function_name(args.type, operation_type) + "(current_uid, original_question, generated_questions_test)")
                # Check if the generated question is of the desired difficulty, if not so try with the next example
                if '_2' in operation_type and generated_question['operation_difficulty'] == 2:
                    found = True
                elif '_3' in operation_type and generated_question['operation_difficulty'] == 3:
                    found = True
                elif '_2' not in operation_type and '_3' not in operation_type:
                    found = True
            print("Generated!")
            generated_questions_test.append(generated_question)
            current_uid += 1

        # Save state in temp file
        with open("dataset/temp_state.json", "w") as json_file:
            data = {"type": args.type, "n": args.n, "operation_type": operation_type, "train_iter_position": candidates_train_iter_position, "test_iter_position": candidates_test_iter_position}
            json.dump(data, json_file, indent=2, ensure_ascii=False)
        # Save generated questions
        with open("dataset/train_generated.json", "w") as json_file:
            json.dump(generated_questions_train, json_file, indent=2, ensure_ascii=False)
        with open("dataset/test_generated.json", "w") as json_file:
            json.dump(generated_questions_test, json_file, indent=2, ensure_ascii=False)
        # Save updated uid
        with open("entities_and_properties/Generated_questions.json", "r") as json_file:
            json_data = json.load(json_file)
            json_data["current_max_uid"] = current_uid
        with open("entities_and_properties/Generated_questions.json", "w") as json_file:
            json.dump(json_data, json_file, indent=2, ensure_ascii=False)
    os.remove("dataset/temp_state.json")