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

from typing import List, Dict, Any
from enum import Enum
import json
import argparse
import re

# These templates identify the type of template and the position of the associated example question
templateTypes = {
    "simple_question_left": 0,
    "right_subgraph": 1,
    "right_subgraph_2": 2, # "left_subgraph" is identical
    "center": 3,
    "center_2": 4,
    "rank": 5,
    "rank_2": 6,
    "string_matching_simple_contains_word": 7, # There are 2 templates in one type
    "statement_property": 8, # There are 2 templates in one type
    "statement_property_2": 9, # There are 2 templates in one type
    "count": 10,
    "count_2": 11,
    "simple_question_right": 12,
    "string_matching_type_relation_contains_word": 13, # There are 2 templates in one type
    "two_intentions_right_subgraph": 14,
    "unknown": 15,
    "unknown_2": 16
}

# Types of implemented operations. Not all of them are implemented for every template, for instance "generic" and "generic_2" are available only
# for "simple_question_left" template
class OperationTypes(Enum):
    entity_example = "entity_example"
    entity = "entity"
    relation = "relation"
    generic = "generic"
    entity_2 = "entity_2"
    relation_2 = "relation_2"
    entity_3 = "entity_3"
    relation_3 = "relation_3"
    generic_2 = "generic_2"

# Get only current uid, without loading all generated questions, since they would be useless
def get_current_max_uid() -> int:
    with open("entities_and_properties/Generated_questions.json", "r") as json_file:
        json_data = json.load(json_file)
    return json_data["current_max_uid"]

# Check if questions has already been generated and add only unique ones, updating uids
def add_unique_generated_questions(generated_questions: List[Dict[str, Any]], old_generated_questions: List[Dict[str, Any]], current_uid: int):
    for i, generated_question in enumerate(generated_questions):
            if not questions_generator.check_question_has_been_generated(generated_question['sparql_wikidata'], old_generated_questions):
                old_generated_questions.append(generated_question)
            else:
                current_uid -= 1
                for next_questions_index in range(i, len(generated_questions)):
                    generated_questions[next_questions_index]['uid'] -= 1

# Load generated questions, add new questions, update uuid and save all
def update_generated_questions_file(current_uid: int, generated_questions: List[Dict[str, Any]]):
    with open("entities_and_properties/Generated_questions.json", "r") as json_file:
        json_data = json.load(json_file)
        # Check if any of the generated questions has already been saved in JSON file
        old_generated_questions = json_data["generated_questions"]
        add_unique_generated_questions(generated_questions, old_generated_questions, current_uid)
        json_data["current_max_uid"] = current_uid
    with open("entities_and_properties/Generated_questions.json", "w") as json_file:
        json.dump(json_data, json_file, indent=2, ensure_ascii=False)

# Load question template from template number
def load_question_template(template_position: int) -> Dict[str, Any]:
    with open("entities_and_properties/Examples_templates.json", "r") as json_file:
        return json.load(json_file)[template_position]

# Empty generated questions file and move generated questions into official training or test set
def move_generated_questions_to_dataset(train: bool):
    # Load generated questions
    with open("entities_and_properties/Generated_questions.json", "r") as json_file:
        json_data = json.load(json_file)
    generated_questions = json_data["generated_questions"]
    json_data["generated_questions"] = []
    # Empty generated questions file
    with open("entities_and_properties/Generated_questions.json", "w") as json_file:
        json.dump(json_data, json_file, indent=2, ensure_ascii=False)
    # Update dataset with generated questions
    if train:
        with open("dataset/train.json", "r") as json_file:
            json_data = json.load(json_file)
        # In this case uid value is not relevant since I don't need it, so any value is correct
        add_unique_generated_questions(generated_questions, json_data, 0)
        with open("dataset/train.json", "w") as json_file:
            json.dump(json_data, json_file, indent=2, ensure_ascii=False)
    else:
        with open("dataset/test.json", "r") as json_file:
            json_data = json.load(json_file)
        # In this case uid value is not relevant since I don't need it, so any value is correct
        add_unique_generated_questions(generated_questions, json_data, 0)
        with open("dataset/test.json", "w") as json_file:
            json.dump(json_data, json_file, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # Define expected terminal arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", help="type of generated questions template", type=str, default="simple_question_left")
    parser.add_argument("--op", help="operation to execute", type=str, default="entity")
    parser.add_argument("--n", help="number of generated examples", type=int, default=5)
    args = parser.parse_args()

    # Execute functions according to selected terminal arguments
    if args.type == "move_to_train":
        move_generated_questions_to_dataset(True)
    elif args.type == "move_to_test":
        move_generated_questions_to_dataset(False)
    else:
        # Load questions generation common data
        current_uid = get_current_max_uid()
        if args.type in templateTypes:
            question_template = load_question_template(templateTypes[args.type])
        else:
            raise NotImplementedError('Not implemented this type of template')
        # Generate questions according to selected terminal arguments
        generated_questions = []
        for n in range(args.n):
            # There could be variations of the same template that are identified with a number, if they are present get the module name removing that number
            template_variation = re.findall(r'(_\d+)', args.type)
            if len(template_variation) > 0:
                module_name = args.type.replace(template_variation[-1], "")
            else:
                module_name = args.type
            # Get function call from input arguments, assuming that arguments and template remain the same
            function = args.type + "_" + args.op + "_generation"
            try:
                exec("generated_questions.append(" + module_name + "." + args.type + "_" + args.op + "_generation(current_uid, question_template, generated_questions))")
            except AttributeError as e:
                # If AttributeError refers to a call of a wrong function due to wrong input arguments, ignore original message and raise a specific error
                error_message = str(e)
                if function in error_message:
                    raise SystemExit('Not implemented this type of operation')
                else:
                    raise
            print("Generated!")
            current_uid += 1
        update_generated_questions_file(current_uid, generated_questions)
