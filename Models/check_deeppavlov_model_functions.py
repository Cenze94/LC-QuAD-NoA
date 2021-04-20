from SPARQLWrapper import SPARQLWrapper, JSON
from SPARQLWrapper.SPARQLExceptions import QueryBadFormed, EndPointInternalError
from typing import List, Any, Dict, Iterator
from urllib.error import HTTPError
import json
import datetime
import time
import re
import requests
import ast

# Support function to manage "Too many requests" Wikidata server error
def get_delay(date):
    try:
        date = datetime.datetime.strptime(date, '%a, %d %b %Y %H:%M:%S GMT')
        timeout = int((date - datetime.datetime.now()).total_seconds())
    except ValueError:
        timeout = int(date)
    return timeout

# Support function to manage "Too many requests" Wikidata server error
def make_request(wikidata_url, headers = None) -> requests.Response:
    if headers:
        r = requests.get(wikidata_url, headers)
    else:
        r = requests.get(wikidata_url)
    if r.status_code == 429:
        timeout = get_delay(r.headers['retry-after'])
        print('Timeout {} m {} s'.format(timeout // 60, timeout % 60))
        time.sleep(timeout)
        make_request(wikidata_url, headers)
    else:
        return r

# Execute SPARQL query and get results
def get_sparql_query_results(sparql_query: str) -> List[Any]:
    endpoint_url = "https://query.wikidata.org/sparql"
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query()
        return results.convert()
    except QueryBadFormed:
        return []
    except EndPointInternalError:
        return []
    except HTTPError as e:
        # Manage "Too Many Requests" error
        timeout = get_delay(e.headers['retry-after'])
        print('Timeout {} m {} s'.format(timeout // 60, timeout % 60))
        time.sleep(timeout)
        return get_sparql_query_results(sparql_query)

# Get entity name using wbgetentities&ids Wikidata service. If entity is not an entity or a propriety, return an empty string
def get_entity_name_from_wikidata_id(entity_id: str) -> str:
    if len(re.findall(r'Q(\d+)', entity_id)) == 0 and len(re.findall(r'P(\d+)', entity_id)) == 0:
        return ""
    data = make_request('https://www.wikidata.org/w/api.php?action=wbgetentities&ids=' + entity_id + '&format=json').json()
    try:
        return data['entities'][entity_id]['labels']['en']['value'].replace("_", " ")
    except KeyError:
        return ""

# Check if input element is an entity, and if so return the corresponding English label
def convert_entity_qid_to_label(element: str) -> str:
    if len(re.findall(r"^Q\d+$", element)) > 0:
        # Element is an entity, return its label
        return get_entity_name_from_wikidata_id(element)
    else:
        # Element is a number, date or another type of string, so return without modifications
        return element

# Create a new JSON file without questions that don't have an answer, and add all possible answers as additional data
def filter_test_questions():
    with open("data/LC-QuAD_2_test.json", "r") as json_file:
        json_data = json.load(json_file)
    filtered_json_data = []
    for element in json_data:
        print(element['sparql_wikidata'])
        results = get_sparql_query_results(element['sparql_wikidata'])
        # There are some questions considered malformed, which have to be skipped and so an empty list is returned. The same is done with
        # queries that cause an Internal Server Error
        if len(results) > 0:
            if 'boolean' in results:
                element['answers'] = [results['boolean']]
            else:
                results = results['results']['bindings']
                # Check if there is at least one English answer in cases different from count, otherwise check if the obtained number is different from 0
                if len(results) > 0 and ((element['template_id'] != "Count_1" and element['template_id'] != "Count_2") or \
                    results['results']['bindings'][0]['value']['value'] != "0"):
                    # Save all answers for a maximum of 50 elements, since getting labels is a slow process and there could be questions with a lot of possible answers
                    # (I found one with more than 14,000 answers); there could be the possibility of not considering the answer proposed by the model, but I think it's
                    # improbable since the answers order of the same query usually doesn't change much
                    answers = []
                    for i in range(min(50, len(results))):
                        answer = convert_entity_qid_to_label(next(iter(results[i].values()))['value'].split("/")[-1])
                        # Answer can be without an English label
                        if answer:
                            answers.append(answer)
                        # Check if there is a second answer and has an English label, if so add to answers list (DeepPavlov always returns only a single answer)
                        if not element['subgraph'] or element['subgraph'] == "two intentions right subgraph":
                            answer = convert_entity_qid_to_label(next(answer_iter)['value'].split("/")[-1])
                            # Answer can be without an English label
                            if answer:
                                answers.append(answer)
                    if not answers:
                        print("No English answers for element: " + str(element))
                    else:
                        element['answers'] = answers
            filtered_json_data.append(element)
    with open("data/LC-QuAD_2_test_filtered.json", "w") as json_file:
        json.dump(filtered_json_data, json_file, indent=2, ensure_ascii=False)

# Add answers to answerable questions of a specific JSON file, for example if that passage has not been done during file construction. Unanswerable questions are managed
def add_answers_to_file(filename: str):
    # Open file
    with open(filename + ".json", "r") as json_file:
        json_data = json.load(json_file)
    for index, question in enumerate(json_data):
        if index % 50 == 0 and index != 0:
            print(index)
        if "has_answer" in question and question['has_answer'] == False:
            # Question doesn't have an answer, so save an empty "answers" list
            question['answers'] = []
        else:
            results = get_sparql_query_results(question['sparql_wikidata'])
            # There are some questions considered malformed, which have to be skipped and so an empty list is returned. The same is done with queries that cause an
            # Internal Server Error
            if len(results) > 0:
                if 'boolean' in results:
                    question['answers'] = [results['boolean']]
                else:
                    results = results['results']['bindings']
                    # Check if there is at least one answer
                    if len(results) > 0:
                        # Save all answers for a maximum of 50 elements, since getting labels is a slow process and there could be questions with a lot of possible answers
                        # (I found one with more than 14,000 answers); there could be the possibility of not considering the answer proposed by the model, but I think it's
                        # improbable since the answers order of the same query usually is more or less the same
                        answers = []
                        for i in range(min(50, len(results))):
                            answer_iter = iter(results[i].values())
                            answer = convert_entity_qid_to_label(next(answer_iter)['value'].split("/")[-1])
                            # Answer can be without an English label
                            if answer:
                                answers.append(answer)
                            # Check if there is a second answer and has an English label, if so add to answers list (DeepPavlov always returns only a single answer)
                            if not question['subgraph'] or question['subgraph'] == "two intentions right subgraph":
                                answer = convert_entity_qid_to_label(next(answer_iter)['value'].split("/")[-1])
                                # Answer can be without an English label
                                if answer:
                                    answers.append(answer)
                        if not answers:
                            print("No English answers for element: " + str(question))
                        question['answers'] = answers
                    else:
                        question['answers'] = []
    with open(filename + ".json", "w") as json_file:
        json.dump(json_data, json_file, indent=2, ensure_ascii=False)

# If input element is a date, change format to Wikidata one
def transform_date(element: str) -> str:
    if len(re.findall(r'^\d{2}\s\w+\s\d+', element)) > 0:
        element_components = element.split()
        # Transform month
        monts_dict = {"January": "01", "February": "02", "March": "03", "April": "04", "May": "05", "June": "06", "July": "07", "August": "08", \
            "September": "09", "October": "10", "November": "11", "December": "12"}
        element_components[1] = monts_dict[element_components[1]]
        # If there is "BCE" in date string add a "-" at the beginning of the returned result and decrease year value of 1
        if " BCE" in element:
            # Save initial zeroes, which are eliminated during int cast
            pos = 0
            zeroes = ""
            while pos < len(element_components[2]) and element_components[2][pos] == "0":
                zeroes += "0"
            element_components[2] = "-" + zeroes + str(int(element_components) - 1)
        return element_components[2] + "-" + element_components[1] + "-" + element_components[0] + "T00:00:00Z"
    else:
        return element

# Check if a group of lines refers to the same question, verifying if they are followed by a "-|-" line: if so take only the last line. If there are
# empty groups (i. e. there is marker immediately after another marker) add an empty line
def get_correct_file_list(original_list: List[Any]) -> List[Any]:
    filtered_list = []
    previous_line = None
    for element in original_list:
        if isinstance(element, str) and element == '-|-':
            # Marker
            if previous_line == None:
                # Empty group, add an empty line
                filtered_list.append("")
            else:
                # Add the last line
                filtered_list.append(previous_line)
                previous_line = None
        else:
            # Group line, save in previous_line
            previous_line = element
    return filtered_list

# Get entities and values from true SPARQL query, maintaining the correct order
def get_true_entities(sparql_query: str) -> List[str]:
    matches_tuples = re.findall(r"wd:(Q\d*)|'(.*?)'", sparql_query)
    # In questions with the check of contained strings there is also the 'en' check for label at the end of the query, which is not an entity and so must be deleted
    if 'lang(' in sparql_query.lower():
        del matches_tuples[-1]
    # Since there is an "or" ("|") in regex, the result of findall is a list of tuples, where every tuple has an empty element and another with the result. As a
    # consequence every tuple must be converted in the not empty element
    matches = []
    for element in matches_tuples:
        if len(element[0]) > 0:
            matches.append(element[0])
        elif len(element[1]) > 0:
            matches.append(element[1])
    return matches

# Check if question is correct using paper method, i. e. if SPARQL entities and values correspond to gold answers (I assume that also templates must be correct,
# although there is no mention of it in the paper)
def get_question_paper_correctness(element: Dict[str, Any], paper_correctness_data: List[Iterator]) -> bool:
    # Get question elements from files to find model query entities
    model_template = next(paper_correctness_data[0])
    model_queries_candidates = next(paper_correctness_data[1])
    model_candidate_outputs_lists = next(paper_correctness_data[2])
    model_candidate_outputs = next(paper_correctness_data[3])
    model_answer_index = next(paper_correctness_data[4])
    # If DeepPavlov model doesn't found any possible answer for question, then the saved index is '-1'
    if model_answer_index > -1:
        # Get the right candidate with the index
        model_candidate_outputs = model_candidate_outputs[model_answer_index]
        # Find the indexes of candidate elements from their modified names
        model_candidate_list_indexes = []
        for i, candidate_component in enumerate(model_candidate_outputs):
            if isinstance(candidate_component, str) and "|" in candidate_component:
                model_candidate_list_indexes.append(candidate_component.split("|")[0].split("§"))
        # Find candidate elements, ignoring properties
        model_candidate_list_elements = []
        for indexes in model_candidate_list_indexes:
            candidate_element = model_candidate_outputs_lists[int(indexes[0])][int(indexes[1])][int(indexes[2])]
            if isinstance(candidate_element, int) or len(re.findall(r"/P\d*$", candidate_element)) == 0:
                model_candidate_list_elements.append(candidate_element)
        # Find query, since all indexes in "model_candidate_list_indexes" differ only for the last index and I don't need it in this case, use only the first
        # index tuple
        index_tuple = model_candidate_list_indexes[0]
        query_data = model_queries_candidates[int(index_tuple[0])]
        query = query_data[0]
        query_values = query_data[1]
        # Load entities
        model_entities = []
        for triple in query:
            # The first and the third element of the triple may be an entity, analyse them
            # Check if triple element is an entity
            if len(re.findall(r"Q\d+$", triple[0])) > 0:
                model_entities.append(triple[0].split("/")[-1])
            # Check if triple element is an entity
            if len(re.findall(r"Q\d+$", triple[2])) > 0:
                model_entities.append(triple[2].split("/")[-1])
        # Add FILTER values
        for value_tuple in query_values:
            if value_tuple[1] != "qualifier":
                model_entities.append(value_tuple[1])
        true_entities = get_true_entities(element['sparql_wikidata'])
        # Two templates have the triples inverted respect to the corresponding original templates, so in these cases verify only the modified entities order
        model_entities_inverted = []
        entities_inverted_templates = ["SELECT ?ent WHERE { ?ent wdt:P31 wd:T1 . wd:E1 wdt:R1 ?ent }",
                                       "SELECT ?ent WHERE { ?ent wdt:P31 wd:T1 . ?ent wdt:R1 wd:E1 }"]
        for i, template in enumerate(entities_inverted_templates):
            entities_inverted_templates[i] = template.lower()
        if model_template in entities_inverted_templates:
            model_entities_inverted = model_entities[::-1]
        """if element['template_id'] == "statement_property_2":
            print(element['NNQT_question'])
            print(model_template)
            print(model_entities)
            print(model_entities_inverted)
            print(true_entities)
            print(str(model_entities == true_entities or model_entities_inverted == true_entities))"""
        if model_entities == true_entities or model_entities_inverted == true_entities:
            return True
    return False

# Utility function for single question statistics update, because the question could not be present in the predictions test set, and so in that case
# statistics must not be updated
def update_question_statistics(statistics: Dict[str, Dict[str, Any]], element: Dict[str, Any], template_name: str, model_answer: str,
question_is_predicted_answerable: bool, index: int, answers: List[Any], question_is_right_paper_calculation: bool):
    # Check model answer and update the corresponding values, both general and specific to difficulty and operation type ones
    if "has_answer" in element and element['has_answer'] == False:
        # Update the not answerable data
        if model_answer == "Not Found":
            statistics[template_name]["operation types statistics"][element['operation_type'] + " " + str(element['operation_difficulty']) + " not found"] += 1
            statistics[template_name]["not found not answerable"] += 1
            statistics["total"]["operation types statistics"][element['operation_type'] + " " + str(element['operation_difficulty']) + " not found"] += 1
            statistics["total"]["not found not answerable"] += 1
        else:
            statistics[template_name]["operation types statistics"][element['operation_type'] + " " + str(element['operation_difficulty']) + " wrong answers"] += 1
            statistics[template_name]["wrong answers not answerable"] += 1
            statistics["total"]["operation types statistics"][element['operation_type'] + " " + str(element['operation_difficulty']) + " wrong answers"] += 1
            statistics["total"]["wrong answers not answerable"] += 1
        if question_is_right_paper_calculation:
            statistics[template_name]["operation types statistics"][element['operation_type'] + " " + str(element['operation_difficulty']) + " right answers paper"] += 1
            statistics[template_name]["right answers paper not answerable"] += 1
            statistics["total"]["operation types statistics"][element['operation_type'] + " " + str(element['operation_difficulty']) + " right answers paper"] += 1
            statistics["total"]["right answers paper not answerable"] += 1
        else:
            statistics[template_name]["operation types statistics"][element['operation_type'] + " " + str(element['operation_difficulty']) + " wrong answers paper"] += 1
            statistics[template_name]["wrong answers paper not answerable"] += 1
            statistics["total"]["operation types statistics"][element['operation_type'] + " " + str(element['operation_difficulty']) + " wrong answers paper"] += 1
            statistics["total"]["wrong answers paper not answerable"] += 1
        if not question_is_predicted_answerable and model_answer != "Not Found":
            statistics[template_name]["operation types statistics"][element['operation_type'] + " " + str(element['operation_difficulty']) + " right prediction wrong answer number"] += 1
            statistics[template_name]["right prediction wrong answer number not answerable"] += 1
            statistics["total"]["operation types statistics"][element['operation_type'] + " " + str(element['operation_difficulty']) + " right prediction wrong answer number"] += 1
            statistics["total"]["right prediction wrong answer number not answerable"] += 1
        elif question_is_predicted_answerable and model_answer == "Not Found":
            statistics[template_name]["operation types statistics"][element['operation_type'] + " " + str(element['operation_difficulty']) + " wrong prediction right answer number"] += 1
            statistics[template_name]["wrong prediction right answer number not answerable"] += 1
            statistics["total"]["operation types statistics"][element['operation_type'] + " " + str(element['operation_difficulty']) + " wrong prediction right answer number"] += 1
            statistics["total"]["wrong prediction right answer number not answerable"] += 1
        statistics[template_name]["operation types statistics"][element['operation_type'] + " " + str(element['operation_difficulty']) + " total number"] += 1
        statistics[template_name]["total number not answerable"] += 1
        statistics["total"]["operation types statistics"][element['operation_type'] + " " + str(element['operation_difficulty']) + " total number"] += 1
        statistics["total"]["total number not answerable"] += 1
    elif template_name == "without english answers":
        # This is a particular case since only original questions can be here
        if model_answer == "Not Found":
            statistics[template_name]["not found"] += 1
            statistics["total"]["not found"] += 1
        else:
            statistics[template_name]["wrong answers"] += 1
            statistics["total"]["wrong answers"] += 1
        if question_is_right_paper_calculation:
            statistics[template_name]["right answers paper"] += 1
            statistics["total"]["right answers paper"] += 1
        else:
            statistics[template_name]["wrong answers paper"] += 1
            statistics["total"]["wrong answers paper"] += 1
        if not question_is_predicted_answerable and model_answer != "Not Found":
            statistics[template_name]["right prediction wrong answer number"] += 1
            statistics["total"]["right prediction wrong answer number"] += 1
        elif question_is_predicted_answerable and model_answer == "Not Found":
            statistics[template_name]["wrong prediction right answer number"] += 1
            statistics["total"]["wrong prediction right answer number"] += 1
    else:
        # Update the answerable data
        if model_answer == "Not Found":
            statistics[template_name]["not found answerable"] += 1
            statistics["total"]["not found answerable"] += 1
        elif model_answer in answers:
            statistics[template_name]["right answers answerable"] += 1
            statistics["total"]["right answers answerable"] += 1
            if not question_is_predicted_answerable:
                statistics[template_name]["wrong prediction right answer number answerable"] += 1
                statistics["total"]["wrong prediction right answer number answerable"] += 1
        else:
            statistics[template_name]["wrong answers answerable"] += 1
            statistics["total"]["wrong answers answerable"] += 1
        if question_is_right_paper_calculation:
            statistics[template_name]["right answers paper answerable"] += 1
            statistics["total"]["right answers paper answerable"] += 1
        else:
            statistics[template_name]["wrong answers paper answerable"] += 1
            statistics["total"]["wrong answers paper answerable"] += 1
        if question_is_predicted_answerable and model_answer == "Not Found":
            statistics[template_name]["right prediction wrong answer number answerable"] += 1
            statistics["total"]["right prediction wrong answer number answerable"] += 1
        statistics[template_name]["total number answerable"] += 1
        statistics["total"]["total number answerable"] += 1
    # Update the general data
    statistics[template_name]["total number"] += 1
    statistics["total"]["total number"] += 1
    statistics[template_name]["answers"].append({"question index": index, "answer": model_answer, "prediction": str(question_is_predicted_answerable == 1), "true answers": answers})

# Get statistics from dataset file (JSON format) and the corresponding answers file (txt format and containing lists in "['content']" format),
# saving all in statistics file (JSON format)
def save_model_accuracy_statistics(dataset_filename: str, answers_filename: str, statistics_filename: str, paper_correctness_filenames: List[str] = None,
predictions_filename = None, predictions_test_set = None):
    # Open dataset file
    with open(dataset_filename + ".json", "r") as json_file:
        json_data = json.load(json_file)
    # Open answers file
    with open(answers_filename + ".txt", "r") as answers_file:
        model_answers = []
        for answer in answers_file:
            # Don't consider "['" and "']\n" characters
            model_answers.append(answer[2:-3])
    # Load files data for paper correctness calculations if exist
    if paper_correctness_filenames:
        paper_correctness_data = []
        # Load model templates
        with open(paper_correctness_filenames[0] + ".txt", "r") as templates_file:
            model_templates = []
            for template in templates_file:
                # Don't consider "\n" character, every line is a single string
                model_templates.append(template[:-1])
            paper_correctness_data.append(iter(get_correct_file_list(model_templates)))
        # Load model candidates queries
        with open(paper_correctness_filenames[1] + ".txt", "r") as queries_file:
            model_queries = []
            for query in queries_file:
                if query[:-1] == '-|-':
                    # Don't consider "\n" character, check if string is the marker to not invoke "literal_eval"
                    model_queries.append(query[:-1])
                else:
                    # Don't consider "\n" character, parse list from string
                    model_queries.append(ast.literal_eval(query[:-1]))
            paper_correctness_data.append(iter(get_correct_file_list(model_queries)))
        # Load candidate outputs complete lists
        with open(paper_correctness_filenames[2] + ".txt", "r") as lists_file:
            candidate_outputs_lists = []
            for element in lists_file:
                if element[:-1] == '-|-':
                    # Don't consider "\n" character, check if string is the marker to not invoke "literal_eval"
                    candidate_outputs_lists.append(element[:-1])
                else:
                    # Don't consider "\n" character, parse list from string
                    candidate_outputs_lists.append(ast.literal_eval(element[:-1]))
            paper_correctness_data.append(iter(get_correct_file_list(candidate_outputs_lists)))
        # Load candidate outputs
        with open(paper_correctness_filenames[3] + ".txt", "r") as candidates_file:
            candidate_outputs = []
            for candidate in candidates_file:
                if candidate[:-1] == '-|-':
                    # Don't consider "\n" character, check if string is the marker to not invoke "literal_eval"
                    candidate_outputs.append(candidate[:-1])
                else:
                    # Don't consider "\n" character, parse list from string
                    candidate_outputs.append(ast.literal_eval(candidate[:-1]))
            paper_correctness_data.append(iter(get_correct_file_list(candidate_outputs)))
        # Load answer indexes
        with open(paper_correctness_filenames[4] + ".txt", "r") as answers_file:
            answer_indexes = []
            for answer_index in answers_file:
                if answer_index[:-1] == "-":
                    # There aren't answers, save a "-1" value to maintain the same type
                    answer_indexes.append(-1)
                elif answer_index[:-1] == '-|-':
                    answer_indexes.append(answer_index[:-1])
                elif answer_index != "\n":
                    # Don't consider "\n" character and get directly the int value
                    answer_indexes.append(int(answer_index[:-1]))
            paper_correctness_data.append(iter(get_correct_file_list(answer_indexes)))
    if predictions_filename:
        # Load predictions
        with open(predictions_filename + ".txt", "r") as predictions_file:
            predictions_data_iter = iter(ast.literal_eval(predictions_file.readline()))
        with open(predictions_test_set + ".json", "r") as predictions_test_set_file:
            predictions_test_set_data = json.load(predictions_test_set_file)
    # Define template types subdivision and identifiers. The key is "subgraph", if there is only one option then the value is template name, otherwise the value is
    # a dictionary containing "template_id" values as keys and the corresponding template names
    templates = {
        "simple question left": "simple question left",
        "right-subgraph": {1: "right subgraph", 2: "right subgraph 2"},
        "left-subgraph": "right subgraph 2",
        "center": {"1.2": "center", "1.1": "center 2"},
        "rank": {"Rank1": "rank", "Rank2": "rank 2"},
        "string matching simple contains word": "string matching simple contains word",
        "statement_property": {"statement_property_2": "statement property", "statement_property_1": "statement property 2", "Count_1": "count", "Count_2": "count 2"},
        "simple question right": "simple question right",
        "string matching type + relation contains word": "string matching type relation contains word",
        "two intentions right subgraph": "two intentions right subgraph",
        "unknown": {3: "unknown", 2: "unknown 2"},
        "boolean with filter": "boolean with filter",
        "boolean one_hop right subgraph": "boolean one-hop right subgraph",
        "boolean double one_hop right subgraph": "boolean double one_hop right subgraph"
    }
    # Statistics container, initialize a specific key for questions that in the dataset don't have English answers (but have answers in a different language), since
    # they are a special case, where the correct answer is considered to be "not found".
    statistics = {"without english answers": {"not found": 0, "wrong answers": 0, "right answers paper": 0, "wrong answers paper": 0, "right prediction wrong answer number": 0, \
        "wrong prediction right answer number": 0, "total number": 0, "answers": []},
        "total": {"right answers answerable": 0, "not found answerable": 0, "wrong answers answerable": 0, "right answers paper answerable": 0, \
        "wrong answers paper answerable": 0, "right prediction wrong answer number answerable": 0, "wrong prediction right answer number answerable": 0, \
        "total number answerable": 0, "operation types statistics": {"entity 1 not found": 0, "entity 1 wrong answers": 0, "entity 1 right answers paper": 0, \
        "entity 1 wrong answers paper": 0, "entity 1 right prediction wrong answer number": 0, "entity 1 wrong prediction right answer number": 0, \
        "entity 1 total number": 0, "relation 1 not found": 0, "relation 1 wrong answers": 0, "relation 1 right answers paper": 0, "relation 1 wrong answers paper": 0, \
        "relation 1 right prediction wrong answer number": 0, "relation 1 wrong prediction right answer number": 0, "relation 1 total number": 0, \
        "entity 2 not found": 0, "entity 2 wrong answers": 0, "entity 2 right answers paper": 0, "entity 2 wrong answers paper": 0, \
        "entity 2 right prediction wrong answer number": 0, "entity 2 wrong prediction right answer number": 0, "entity 2 total number": 0, "relation 2 not found": 0, \
        "relation 2 wrong answers": 0, "relation 2 right answers paper": 0, "relation 2 wrong answers paper": 0, "relation 2 right prediction wrong answer number": 0, \
        "relation 2 wrong prediction right answer number": 0, "relation 2 total number": 0, "entity 3 not found": 0, "entity 3 wrong answers": 0, \
        "entity 3 right answers paper": 0, "entity 3 wrong answers paper": 0, "entity 3 right prediction wrong answer number": 0, "entity 3 wrong prediction right answer number": 0, \
        "entity 3 total number": 0, "relation 3 not found": 0, "relation 3 wrong answers": 0, "relation 3 right answers paper": 0, "relation 3 wrong answers paper": 0, \
        "relation 3 right prediction wrong answer number": 0, "relation 3 wrong prediction right answer number": 0, "relation 3 total number": 0}, \
        "not found not answerable": 0, "wrong answers not answerable": 0, "right answers paper not answerable": 0, "wrong answers paper not answerable": 0, \
        "right prediction wrong answer number not answerable": 0, "wrong prediction right answer number not answerable": 0, "total number not answerable": 0, "total number": 0}}
    model_answers_iter = iter(model_answers)
    # Update statistics
    for index, element in enumerate(json_data):
        answers = element['answers']
        # Find the correct "statistics" key
        if (not "has_answer" in element or element['has_answer'] == True) and not answers:
            # If there aren't English answers and the question is answerable, update "without english answers" case
            template_name = "without english answers"
        else:
            # Unknown questions has an empty list as "subgraph", but I can't use it as a key for templates dictionary, so I have to manage this case here
            if isinstance(element['subgraph'], list):
                template_name = templates["unknown"]
            else:
                template_name = templates[element['subgraph']]
            if isinstance(template_name, dict):
                template_name = template_name[element['template_id']]
            if template_name not in statistics:
                statistics[template_name] = {"right answers answerable": 0, "not found answerable": 0, "wrong answers answerable": 0, "right answers paper answerable": 0, \
                    "wrong answers paper answerable": 0, "right prediction wrong answer number answerable": 0, "wrong prediction right answer number answerable": 0, \
                    "total number answerable": 0, "operation types statistics": {"entity 1 not found": 0, "entity 1 wrong answers": 0, "entity 1 right answers paper": 0, \
                    "entity 1 wrong answers paper": 0, "entity 1 right prediction wrong answer number": 0, "entity 1 wrong prediction right answer number": 0, \
                    "entity 1 total number": 0, "relation 1 not found": 0, "relation 1 wrong answers": 0, "relation 1 right answers paper": 0, "relation 1 wrong answers paper": 0, \
                    "relation 1 right prediction wrong answer number": 0, "relation 1 wrong prediction right answer number": 0, "relation 1 total number": 0, \
                    "entity 2 not found": 0, "entity 2 wrong answers": 0, "entity 2 right answers paper": 0, "entity 2 wrong answers paper": 0, \
                    "entity 2 right prediction wrong answer number": 0, "entity 2 wrong prediction right answer number": 0, "entity 2 total number": 0, "relation 2 not found": 0, \
                    "relation 2 wrong answers": 0, "relation 2 right answers paper": 0, "relation 2 wrong answers paper": 0, "relation 2 right prediction wrong answer number": 0, \
                    "relation 2 wrong prediction right answer number": 0, "relation 2 total number": 0, "entity 3 not found": 0, "entity 3 wrong answers": 0, \
                    "entity 3 right answers paper": 0, "entity 3 wrong answers paper": 0, "entity 3 right prediction wrong answer number": 0, "entity 3 wrong prediction right answer number": 0, \
                    "entity 3 total number": 0, "relation 3 not found": 0, "relation 3 wrong answers": 0, "relation 3 right answers paper": 0, "relation 3 wrong answers paper": 0, \
                    "relation 3 right prediction wrong answer number": 0, "relation 3 wrong prediction right answer number": 0, "relation 3 total number": 0}, \
                    "not found not answerable": 0, "wrong answers not answerable": 0, "right answers paper not answerable": 0, "wrong answers paper not answerable": 0, \
                    "right prediction wrong answer number not answerable": 0, "wrong prediction right answer number not answerable": 0, "total number not answerable": 0, "total number": 0, "answers": []}
        model_answer = transform_date(next(model_answers_iter))
        # Check if answer is right according to paper calculation, if filenames are not specified then return False to have a final result of 0
        if paper_correctness_filenames:
            question_is_right_paper_calculation = get_question_paper_correctness(element, paper_correctness_data)
        else:
            question_is_right_paper_calculation = False
        # Check answer predictions if the corresponding filename is defined, and if the question is not present in the predictions test set skip it
        if predictions_filename:
            found = False
            pred_index = 0
            while pred_index < len(predictions_test_set_data) and not found:
                if predictions_test_set_data[pred_index]['uid'] == element['uid']:
                    found = True
                pred_index += 1
            # Update statistics only if the element is present in the predictions test set
            if found:
                question_is_predicted_answerable = next(predictions_data_iter)
                update_question_statistics(statistics, element, template_name, model_answer, question_is_predicted_answerable, index, answers, question_is_right_paper_calculation)
        else:
            question_is_predicted_answerable = False
            update_question_statistics(statistics, element, template_name, model_answer, question_is_predicted_answerable, index, answers, question_is_right_paper_calculation)
    # Calculate accuracy for all cases
    statistics_cases = list(statistics.keys()) + ["total"]
    for key in statistics_cases:
        # Manage "without english answers" special case
        if key == "without english answers" and statistics[key]['total number'] > 0:
            statistics[key]['accuracy'] = statistics[key]['not found'] / statistics[key]['total number']
            statistics[key]['accuracy paper'] = statistics[key]['right answers paper'] / statistics[key]['total number']
            statistics[key]['accuracy with predictions'] = (statistics[key]['not found'] + statistics[key]['right prediction wrong answer number'] - \
                statistics[key]['wrong prediction right answer number']) / statistics[key]['total number']
        elif key != "without english answers":
            statistics[key]['accuracy answerable'] = statistics[key]['right answers answerable'] / statistics[key]['total number answerable']
            statistics[key]['accuracy paper answerable'] = statistics[key]['right answers paper answerable'] / statistics[key]['total number answerable']
            statistics[key]['accuracy with predictions answerable'] = (statistics[key]['right answers answerable'] + statistics[key]['right prediction wrong answer number answerable'] - \
                statistics[key]['wrong prediction right answer number answerable']) / statistics[key]['total number answerable']
            
            if statistics[key]["operation types statistics"]['entity 1 total number'] > 0:
                statistics[key]["operation types statistics"]['accuracy entity 1'] = statistics[key]["operation types statistics"]['entity 1 not found'] / \
                    statistics[key]["operation types statistics"]['entity 1 total number']
                statistics[key]["operation types statistics"]['accuracy paper entity 1'] = statistics[key]["operation types statistics"]['entity 1 right answers paper'] / \
                    statistics[key]["operation types statistics"]['entity 1 total number']
                statistics[key]["operation types statistics"]['accuracy with predictions entity 1'] = (statistics[key]["operation types statistics"]['entity 1 not found'] + \
                    statistics[key]["operation types statistics"]['entity 1 right prediction wrong answer number'] - \
                    statistics[key]["operation types statistics"]['entity 1 wrong prediction right answer number']) / statistics[key]["operation types statistics"]['entity 1 total number']
            else:
                statistics[key]["operation types statistics"]['accuracy entity 1'] = 0
                statistics[key]["operation types statistics"]['accuracy paper entity 1'] = 0
                statistics[key]["operation types statistics"]['accuracy with predictions entity 1'] = 0
            
            if statistics[key]["operation types statistics"]['relation 1 total number'] > 0:
                statistics[key]["operation types statistics"]['accuracy relation 1'] = statistics[key]["operation types statistics"]['relation 1 not found'] / \
                    statistics[key]["operation types statistics"]['relation 1 total number']
                statistics[key]["operation types statistics"]['accuracy paper relation 1'] = statistics[key]["operation types statistics"]['relation 1 right answers paper'] / \
                    statistics[key]["operation types statistics"]['relation 1 total number']
                statistics[key]["operation types statistics"]['accuracy with predictions relation 1'] = (statistics[key]["operation types statistics"]['relation 1 not found'] + \
                    statistics[key]["operation types statistics"]['relation 1 right prediction wrong answer number'] - \
                    statistics[key]["operation types statistics"]['relation 1 wrong prediction right answer number']) / statistics[key]["operation types statistics"]['relation 1 total number']
            else:
                statistics[key]["operation types statistics"]['accuracy relation 1'] = 0
                statistics[key]["operation types statistics"]['accuracy paper relation 1'] = 0
                statistics[key]["operation types statistics"]['accuracy with predictions relation 1'] = 0
            
            if statistics[key]["operation types statistics"]['entity 2 total number'] > 0:
                statistics[key]["operation types statistics"]['accuracy entity 2'] = statistics[key]["operation types statistics"]['entity 2 not found'] / \
                    statistics[key]["operation types statistics"]['entity 2 total number']
                statistics[key]["operation types statistics"]['accuracy paper entity 2'] = statistics[key]["operation types statistics"]['entity 2 right answers paper'] / \
                    statistics[key]["operation types statistics"]['entity 2 total number']
                statistics[key]["operation types statistics"]['accuracy with predictions entity 2'] = (statistics[key]["operation types statistics"]['entity 2 not found'] + \
                    statistics[key]["operation types statistics"]['entity 2 right prediction wrong answer number'] - \
                    statistics[key]["operation types statistics"]['entity 2 wrong prediction right answer number']) / statistics[key]["operation types statistics"]['entity 2 total number']
            else:
                statistics[key]["operation types statistics"]['accuracy entity 2'] = 0
                statistics[key]["operation types statistics"]['accuracy paper entity 2'] = 0
                statistics[key]["operation types statistics"]['accuracy with predictions entity 2'] = 0
            
            if statistics[key]["operation types statistics"]['relation 2 total number'] > 0:
                statistics[key]["operation types statistics"]['accuracy relation 2'] = statistics[key]["operation types statistics"]['relation 2 not found'] / \
                    statistics[key]["operation types statistics"]['relation 2 total number']
                statistics[key]["operation types statistics"]['accuracy paper relation 2'] = statistics[key]["operation types statistics"]['relation 2 right answers paper'] / \
                    statistics[key]["operation types statistics"]['relation 2 total number']
                statistics[key]["operation types statistics"]['accuracy with predictions relation 2'] = (statistics[key]["operation types statistics"]['relation 2 not found'] + \
                    statistics[key]["operation types statistics"]['relation 2 right prediction wrong answer number'] - \
                    statistics[key]["operation types statistics"]['relation 2 wrong prediction right answer number']) / statistics[key]["operation types statistics"]['relation 2 total number']
            else:
                statistics[key]["operation types statistics"]['accuracy relation 2'] = 0
                statistics[key]["operation types statistics"]['accuracy paper relation 2'] = 0
                statistics[key]["operation types statistics"]['accuracy with predictions relation 2'] = 0
            
            if statistics[key]["operation types statistics"]['entity 3 total number'] > 0:
                statistics[key]["operation types statistics"]['accuracy entity 3'] = statistics[key]["operation types statistics"]['entity 3 not found'] / \
                    statistics[key]["operation types statistics"]['entity 3 total number']
                statistics[key]["operation types statistics"]['accuracy paper entity 3'] = statistics[key]["operation types statistics"]['entity 3 right answers paper'] / \
                    statistics[key]["operation types statistics"]['entity 3 total number']
                statistics[key]["operation types statistics"]['accuracy with predictions entity 3'] = (statistics[key]["operation types statistics"]['entity 3 not found'] + \
                    statistics[key]["operation types statistics"]['entity 3 right prediction wrong answer number'] - \
                    statistics[key]["operation types statistics"]['entity 3 wrong prediction right answer number']) / statistics[key]["operation types statistics"]['entity 3 total number']
            else:
                statistics[key]["operation types statistics"]['accuracy entity 3'] = 0
                statistics[key]["operation types statistics"]['accuracy paper entity 3'] = 0
                statistics[key]["operation types statistics"]['accuracy with predictions entity 3'] = 0
            
            if statistics[key]["operation types statistics"]['relation 3 total number'] > 0:
                statistics[key]["operation types statistics"]['accuracy relation 3'] = statistics[key]["operation types statistics"]['relation 3 not found'] / \
                    statistics[key]["operation types statistics"]['relation 3 total number']
                statistics[key]["operation types statistics"]['accuracy paper relation 3'] = statistics[key]["operation types statistics"]['relation 3 right answers paper'] / \
                    statistics[key]["operation types statistics"]['relation 3 total number']
                statistics[key]["operation types statistics"]['accuracy with predictions relation 3'] = (statistics[key]["operation types statistics"]['relation 3 not found'] + \
                    statistics[key]["operation types statistics"]['relation 3 right prediction wrong answer number'] - \
                    statistics[key]["operation types statistics"]['relation 3 wrong prediction right answer number']) / statistics[key]["operation types statistics"]['relation 3 total number']
            else:
                statistics[key]["operation types statistics"]['accuracy relation 3'] = 0
                statistics[key]["operation types statistics"]['accuracy paper relation 3'] = 0
                statistics[key]["operation types statistics"]['accuracy with predictions relation 3'] = 0
            
            statistics[key]['accuracy not answerable'] = statistics[key]['not found not answerable'] / statistics[key]['total number not answerable']
            statistics[key]['accuracy paper not answerable'] = statistics[key]['right answers paper not answerable'] / statistics[key]['total number not answerable']
            statistics[key]['accuracy with predictions not answerable'] = (statistics[key]['not found not answerable'] + statistics[key]['right prediction wrong answer number not answerable'] - \
                statistics[key]['wrong prediction right answer number not answerable']) / statistics[key]['total number not answerable']
            
            statistics[key]['final accuracy'] = (statistics[key]['right answers answerable'] + statistics[key]["not found not answerable"]) / statistics[key]['total number']
            statistics[key]['final accuracy paper'] = (statistics[key]['right answers paper answerable'] + statistics[key]["right answers paper not answerable"]) / \
                statistics[key]['total number']
            statistics[key]['final accuracy with answerable predictions'] = (statistics[key]['right answers answerable'] + statistics[key]["not found not answerable"] + \
                statistics[key]['right prediction wrong answer number answerable'] - statistics[key]['wrong prediction right answer number answerable']) / statistics[key]['total number']
            statistics[key]['final accuracy with not answerable predictions'] = (statistics[key]['right answers answerable'] + statistics[key]["not found not answerable"] + \
                statistics[key]['right prediction wrong answer number not answerable'] - statistics[key]['wrong prediction right answer number not answerable']) / \
                statistics[key]['total number']
            statistics[key]['final accuracy with all predictions'] = (statistics[key]['right answers answerable'] + statistics[key]["not found not answerable"] + \
                statistics[key]['right prediction wrong answer number answerable'] - statistics[key]['wrong prediction right answer number answerable'] + \
                statistics[key]['right prediction wrong answer number not answerable'] - statistics[key]['wrong prediction right answer number not answerable']) / \
                statistics[key]['total number']
            statistics[key]['final accuracy answerable predictions no I don\'t know'] = (statistics[key]['right answers answerable'] - statistics[key]['wrong prediction right answer number answerable'] + \
                statistics[key]["not found not answerable"]) / (statistics[key]['total number'] - statistics[key]['right prediction wrong answer number answerable'])
            statistics[key]['final accuracy all predictions no I don\'t know'] = (statistics[key]['right answers answerable'] + statistics[key]["not found not answerable"] - \
                statistics[key]['wrong prediction right answer number answerable'] + statistics[key]['right prediction wrong answer number not answerable'] - \
                statistics[key]['wrong prediction right answer number not answerable']) / (statistics[key]['total number'] - statistics[key]['right prediction wrong answer number answerable'])

    with open(statistics_filename + ".json", "w") as json_file:
        json.dump(statistics, json_file, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    #filter_test_questions()
    #add_answers_to_file("data/LC-QuAD_2_test_balanced")
    common_string = "_test_balanced"
    #save_model_accuracy_statistics("data/LC-QuAD_2" + common_string, "output/deeppavlov_answers" + common_string, "data/test_statistics_balanced")
    save_model_accuracy_statistics("data/LC-QuAD_2" + common_string, "output/deeppavlov_answers" + common_string, "data/test_statistics_balanced", \
        ["output/queries_templates" + common_string, "output/queries_candidates" + common_string, "output/candidate_outputs_lists" + common_string, \
        "output/candidate_outputs" + common_string, "output/answers_indexes" + common_string])
    """save_model_accuracy_statistics("data/LC-QuAD_2" + common_string, "output/deeppavlov_answers" + common_string, "data/test_statistics_balanced_with_embeddings_lstm_gold_model", \
        ["output/queries_templates" + common_string, "output/queries_candidates" + common_string, "output/candidate_outputs_lists" + common_string, \
        "output/candidate_outputs" + common_string, "output/answers_indexes" + common_string], "models/lstm_embeddings_gold_model/model_predictions", \
        "data/LC_QuAD_2_test_balanced_with_embeddings")"""