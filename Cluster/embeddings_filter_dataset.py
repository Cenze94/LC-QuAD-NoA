import json
import re
import sqlite3
import ast
import random
from copy import deepcopy
from typing import List, Tuple, Any, Iterator, Dict

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

# Dictionary containing the encodings of all possible templates (encodings values are calculated automatically during the execution)
templates_dp_encoding = {
    "select ?obj where { wd:e1 p:r1 ?s . ?s ps:r1 ?obj . ?s ?p ?x filter(contains(?x, n)) }": {},
    "select ?value where { wd:e1 p:r1 ?s . ?s ps:r1 ?x filter(contains(?x, n)) . ?s ?p ?value }": {},
    "select ?value where { wd:e1 p:r1 ?s . ?s ps:r1 wd:e2 . ?s ?p ?value }": {},
    "select ?obj where { wd:e1 p:r1 ?s . ?s ps:r1 ?obj . ?s ?p wd:e2 }": {},
    "select (count(?obj) as ?value ) { wd:e1 wdt:r1 ?obj }": {},
    "select ?ent where { ?ent wdt:p31 wd:t1 . ?ent wdt:r1 ?obj } order by asc(?obj) limit 5": {},
    "select ?ent where { ?ent wdt:p31 wd:t1 . ?ent wdt:r1 ?obj . ?ent wdt:r2 wd:e1 } order by asc(?obj) limit 5": {},
    "select ?ent where { wd:e1 wdt:r1 ?ent }": {},
    "select ?ent where { ?ent wdt:r1 wd:e1 }": {},
    "select ?ent where { wd:e1 wdt:r1 ?ent . wd:e2 wdt:r2 ?ent }": {},
    "select ?ent where { wd:e1 wdt:r1 ?ent . ?ent ?p wd:e2 }": {},
    "select ?ent where { wd:e1 wdt:r1 ?ent . wd:e2 wdt:r2 ?ent . wd:e3 wdt:r3 ?ent }": {},
    "select ?ent where { ?ent wdt:p31 wd:t1 . wd:e1 wdt:r1 ?ent }": {},
    "select ?ent where { ?ent wdt:p31 wd:t1 . ?ent wdt:r1 wd:e1 }": {},
    "select ?ent where { ?ent wdt:p31 wd:t1 . wd:e1 wdt:r1 ?ent . ?ent ?p wd:e2 }": {},
    "select ?ent where { ?ent wdt:p31 wd:t1 . wd:e1 wdt:r1 ?ent . wd:e2 wdt:r2 ?ent }": {},
    "select ?ent where { ?ent wdt:r1 wd:e1 . ?ent wdt:r2 wd:e2 }": {},
    "select ?ent where { ?ent wdt:p31 wd:t1 . ?ent wdt:r1 wd:e1 . ?ent wdt:r2 wd:e2 }": {},
    "select ?ent where { wd:e1 wdt:r1 ?ent . ?ent wdt:r2 wd:e2 . ?ent wdt:r3 wd:e3 }": {},
    "select ?ent where { ?ent wdt:p31 wd:t1 . wd:e1 wdt:r1 ?ent . ?ent wdt:r2 wd:e2 . ?ent wdt:r3 wd:e3 }": {},
    "select ?ent where { ?ent_mid wdt:p31 wd:t1 . ?ent wdt:r1 ?obj . ?ent_mid wdt:r2 ?ent } order by asc(?obj) limit 5": {},
    "select ?ent where { wd:e1 wdt:r1 ?ent_mid . ?ent_mid ps:r1 ?ent . ?ent_mid ?p ?beg } order by asc(?beg) limit 5": {},
    "select ?ent where { wd:e1 p:r1 ?ent_mid . ?ent_mid ps:r1 ?ent . ?ent_mid pq:p580 ?beg } order by asc(?beg) limit 5": {},
    "select ?ent where { wd:e1 p:r1 ?ent_mid . ?ent_mid ps:r1 ?ent . ?ent_mid pq:p585 ?beg } order by asc(?beg) limit 5": {},
    "select ?ent where { wd:e1 p:r1 ?ent_mid . ?ent_mid ps:r1 ?ent . ?ent_mid pq:p582 ?beg } order by desc(?beg) limit 5": {},
    "no_recognized_template": {} # Support template, since questions without an associated template doesn't have embeddings, and as a consequence they will be excluded
}

# Dictionary containing the correspondence map between model templates and original ones
model_templates_map = {
    "select ?obj where { wd:e1 p:r1 ?s . ?s ps:r1 ?obj . ?s ?p ?x filter(contains(?x, n)) }": {"subgraph": "statement_property", "template_id": "statement_property_2"},
    "select ?value where { wd:e1 p:r1 ?s . ?s ps:r1 ?x filter(contains(?x, n)) . ?s ?p ?value }": {"subgraph": "statement_property", "template_id": "statement_property_1"},
    "select ?value where { wd:e1 p:r1 ?s . ?s ps:r1 wd:e2 . ?s ?p ?value }": {"subgraph": "statement_property", "template_id": "statement_property_1"},
    "select ?obj where { wd:e1 p:r1 ?s . ?s ps:r1 ?obj . ?s ?p wd:e2 }": {"subgraph": "statement_property", "template_id": "statement_property_2"},
    "select (count(?obj) as ?value ) { wd:e1 wdt:r1 ?obj }": {"subgraph": "statement_property", "template_id": "Count_1"},
    "select ?ent where { ?ent wdt:p31 wd:t1 . ?ent wdt:r1 ?obj } order by asc(?obj) limit 5": {"subgraph": "rank", "template_id": "Rank1"},
    "select ?ent where { ?ent wdt:p31 wd:t1 . ?ent wdt:r1 ?obj . ?ent wdt:r2 wd:e1 } order by asc(?obj) limit 5": {"subgraph": "rank", "template_id": "Rank2"},
    "select ?ent where { wd:e1 wdt:r1 ?ent }": {"subgraph": "center", "template_id": "1.1"},
    "select ?ent where { ?ent wdt:r1 wd:e1 }": {"subgraph": "center", "template_id": "1.2"},
    "select ?ent where { wd:e1 wdt:r1 ?ent . wd:e2 wdt:r2 ?ent }": {"subgraph": "right-subgraph", "template_id": 1},
    "select ?ent where { wd:e1 wdt:r1 ?ent . ?ent ?p wd:e2 }": {"subgraph": "right-subgraph", "template_id": 1},
    "select ?ent where { wd:e1 wdt:r1 ?ent . wd:e2 wdt:r2 ?ent . wd:e3 wdt:r3 ?ent }": {"subgraph": "right-subgraph", "template_id": 1},
    "select ?ent where { ?ent wdt:p31 wd:t1 . wd:e1 wdt:r1 ?ent }": {"subgraph": "simple question left", "template_id": 2},
    "select ?ent where { ?ent wdt:p31 wd:t1 . ?ent wdt:r1 wd:e1 }": {"subgraph": "simple question right", "template_id": 1},
    "select ?ent where { ?ent wdt:p31 wd:t1 . wd:e1 wdt:r1 ?ent . ?ent ?p wd:e2 }": {"subgraph": "rank", "template_id": "Rank2"},
    "select ?ent where { ?ent wdt:p31 wd:t1 . wd:e1 wdt:r1 ?ent . wd:e2 wdt:r2 ?ent }": {"subgraph": "rank", "template_id": "Rank2"},
    "select ?ent where { ?ent wdt:r1 wd:e1 . ?ent wdt:r2 wd:e2 }": {"subgraph": "right-subgraph", "template_id": 1},
    "select ?ent where { ?ent wdt:p31 wd:t1 . ?ent wdt:r1 wd:e1 . ?ent wdt:r2 wd:e2 }": {"subgraph": "rank", "template_id": "Rank2"},
    "select ?ent where { wd:e1 wdt:r1 ?ent . ?ent wdt:r2 wd:e2 . ?ent wdt:r3 wd:e3 }": {"subgraph": "right-subgraph", "template_id": 1},
    "select ?ent where { ?ent wdt:p31 wd:t1 . wd:e1 wdt:r1 ?ent . ?ent wdt:r2 wd:e2 . ?ent wdt:r3 wd:e3 }": {"subgraph": "right-subgraph", "template_id": 1},
    "select ?ent where { ?ent_mid wdt:p31 wd:t1 . ?ent wdt:r1 ?obj . ?ent_mid wdt:r2 ?ent } order by asc(?obj) limit 5": {"subgraph": "rank", "template_id": "Rank2"},
    "select ?ent where { wd:e1 wdt:r1 ?ent_mid . ?ent_mid ps:r1 ?ent . ?ent_mid ?p ?beg } order by asc(?beg) limit 5": {"subgraph": "statement_property", "template_id": "statement_property_2"},
    "select ?ent where { wd:e1 p:r1 ?ent_mid . ?ent_mid ps:r1 ?ent . ?ent_mid pq:p580 ?beg } order by asc(?beg) limit 5": {"subgraph": "statement_property", "template_id": "statement_property_2"},
    "select ?ent where { wd:e1 p:r1 ?ent_mid . ?ent_mid ps:r1 ?ent . ?ent_mid pq:p585 ?beg } order by asc(?beg) limit 5": {"subgraph": "statement_property", "template_id": "statement_property_2"},
    "select ?ent where { wd:e1 p:r1 ?ent_mid . ?ent_mid ps:r1 ?ent . ?ent_mid pq:p582 ?beg } order by desc(?beg) limit 5": {"subgraph": "statement_property", "template_id": "statement_property_2"},
    "no_recognized_template": {"subgraph": "right-subgraph", "template_id": 1} # Assign a random template because questions without an associated model template don't have embeddings and so are excluded
}

# Prepare one-hot encodings of templates
def prepare_encodings():
    # Ground truth templates
    templates_number = 0
    for subgraph in templates_encoding.keys():
        templates_number += len(templates_encoding[subgraph])
    one_position = 0
    for subgraph in templates_encoding.keys():
        for template_id in templates_encoding[subgraph].keys():
            templates_encoding[subgraph][template_id] = [0] * templates_number
            templates_encoding[subgraph][template_id][one_position] = 1
            one_position += 1
    # DeepPavlov templates
    templates_number = 0
    for subgraph in templates_dp_encoding.keys():
        templates_number += 1
    one_position = 0
    for subgraph in templates_dp_encoding.keys():
        templates_dp_encoding[subgraph] = [0] * templates_number
        templates_dp_encoding[subgraph][one_position] = 1
        one_position += 1

# Get list of DeepPavlov embeddings types in the right order
def get_embeddings_schema_from_query_deeppavlov(paper_correctness_data: List[Iterator], question: Dict[str, Any]) -> List[List[str]]:
    # Get question elements from files to find model query entities
    model_template = next(paper_correctness_data[0])
    model_queries_candidates = next(paper_correctness_data[1])
    model_candidate_outputs_lists = next(paper_correctness_data[2])
    model_candidate_outputs = next(paper_correctness_data[3])
    model_answer_index = next(paper_correctness_data[4])
    # Save model template in question
    if model_template:
        question['deeppavlov_template'] = model_template
    else:
        question['deeppavlov_template'] = "no_recognized_template"
    embeddings_list = []
    """# If DeepPavlov model doesn't found any possible answer for question, then the saved index is '-1'
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
        for triple in query:
            # The first and the third element of the triple may be an entity, and the second one may be a relation
            # Check if triple element is an entity
            if len(re.findall(r"Q\d+$", triple[0])) > 0:
                embeddings_list.append(triple[0].split("/")[-1])
            # Check if triple element is a property
            if len(re.findall(r"P\d+$", triple[1])) > 0:
                embeddings_list.append(triple[1].split("/")[-1])
            # Check if triple element is an entity
            if len(re.findall(r"Q\d+$", triple[2])) > 0:
                embeddings_list.append(triple[2].split("/")[-1])"""

    # Load every candidate query
    for query_data in model_queries_candidates:
        # Get triples
        query = query_data[0]
        queries_list = []
        for triple in query:
            # The first and the third element of the triple may be an entity, and the second one may be a relation
            # Check if triple element is an entity
            if len(re.findall(r"Q\d+$", triple[0])) > 0:
                queries_list.append(triple[0].split("/")[-1])
            # Check if triple element is a property
            if len(re.findall(r"P\d+$", triple[1])) > 0:
                queries_list.append(triple[1].split("/")[-1])
            # Check if triple element is an entity
            if len(re.findall(r"Q\d+$", triple[2])) > 0:
                queries_list.append(triple[2].split("/")[-1])
        embeddings_list.append(queries_list)
    return embeddings_list

# Get list of train embeddings types in the right order
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

def preprocess_questions_file(filename: str, additional_name: str = "", without_dp_embeddings = False):
    with open("data/LC-QuAD_2_" + filename + "_balanced.json", "r") as json_file:
        json_data = json.load(json_file)
    # Load files for test
    paper_correctness_data = []
    # Load model templates
    with open("output/queries_templates_" + filename + "_balanced.txt", "r") as templates_file:
        model_templates = []
        for template in templates_file:
            # Don't consider "\n" character, every line is a single string
            model_templates.append(template[:-1])
        paper_correctness_data.append(iter(get_correct_file_list(model_templates)))
    # Load model candidates queries
    with open("output/queries_candidates_" + filename + "_balanced.txt", "r") as queries_file:
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
    with open("output/candidate_outputs_lists_" + filename + "_balanced.txt", "r") as lists_file:
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
    with open("output/candidate_outputs_" + filename + "_balanced.txt", "r") as candidates_file:
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
    with open("output/answers_indexes_" + filename + "_balanced.txt", "r") as answers_indexes_file:
        answer_indexes = []
        for answer_index in answers_indexes_file:
            if answer_index[:-1] == "-":
                # There aren't answers, save a "-1" value to maintain the same type
                answer_indexes.append(-1)
            elif answer_index[:-1] == '-|-':
                answer_indexes.append(answer_index[:-1])
            elif answer_index != "\n":
                # Don't consider "\n" character and get directly the int value
                answer_indexes.append(int(answer_index[:-1]))
        paper_correctness_data.append(iter(get_correct_file_list(answer_indexes)))
    # Load model answers
    with open("output/deeppavlov_answers_" + filename + "_balanced.txt", "r") as answers_file:
        answers_list = []
        for answer in answers_file:
            # Don't consider "['" and "']\n" characters
            answers_list.append(answer[2:-3])
        paper_correctness_data.append(iter(answers_list))
    # Connect to DB
    conn = sqlite3.connect("../LC-QuAD-NoA/mini-dataset/mini-dataset_embeddings.db")
    c = conn.cursor()
    # Create a new preprocessed set file
    with open("data/LC_QuAD_2_" + filename + "_balanced_with_embeddings" + additional_name + ".json", "w") as set_file:
        set_file.write("[")
    # JSON writing utility variable
    first_element = True
    no_embedding_count = 0
    no_embedding_deeppavlov_count = 0
    for index, question in enumerate(json_data):
        # If question lacks any embedding, exclude from final data
        no_embedding = False
        no_embedding_deeppavlov = False
        if question['question']:
            question['model_question'] = question['question']
        else:
            question['model_question'] = question['NNQT_question']
        if 'has_answer' in question and question['has_answer'] == False:
            question['answerable'] = 0
        else:
            question['answerable'] = 1
        # Save DeepPavlov question
        question['deeppavlov_answer'] = next(paper_correctness_data[5])

        if not without_dp_embeddings:
            # Get DeepPavlov embeddings and template
            embeddings_list_deeppavlov = get_embeddings_schema_from_query_deeppavlov(paper_correctness_data, question)
            """# Exclude questions without embeddings, because they can't be used for LSTM
            if not embeddings_list_deeppavlov:
                no_embedding_deeppavlov_count += 1
                no_embedding_deeppavlov = True
            for i, element in enumerate(embeddings_list_deeppavlov):
                if not no_embedding_deeppavlov:
                    result = find_element_DB(element, c)
                    if result is None:
                        no_embedding_deeppavlov_count += 1
                        no_embedding_deeppavlov = True
                    else:
                        embeddings_list_deeppavlov[i] = result[1:]
            if not no_embedding_deeppavlov:
                # Get LC-QuAD embeddings and template
                # Find embeddings
                embeddings_list = get_embeddings_schema_from_query(question['sparql_wikidata'])
                # Exclude questions without embeddings, because they can't be used for LSTM
                if not embeddings_list:
                    no_embedding_count += 1
                    no_embedding = True
                for i, element in enumerate(embeddings_list):
                    if not no_embedding:
                        result = find_element_DB(element, c)
                        if result is None:
                            no_embedding_count += 1
                            no_embedding = True
                        else:
                            embeddings_list[i] = result[1:]
                if not no_embedding:
                    question['deeppavlov_embeddings'] = embeddings_list_deeppavlov
                    question['embeddings'] = embeddings_list
                    set_file.write(json.dumps(question, indent=2, ensure_ascii=False) + "\n")"""
            
            # Get embeddings for every query candidate
            no_embedding_deeppavlov_list = True
            filtered_embeddings_list_deeppavlov = []
            # Count questions without candidate queries
            if not embeddings_list_deeppavlov:
                no_embedding_deeppavlov_count += 1
            for i, query_candidate in enumerate(embeddings_list_deeppavlov):
                no_embedding_deeppavlov = False
                for j, element in enumerate(query_candidate):
                    # If even just one embedding is missing, discard the query candidate
                    if not no_embedding_deeppavlov:
                        result = find_element_DB(element, c)
                        if result is None:
                            no_embedding_deeppavlov = True
                        else:
                            embeddings_list_deeppavlov[i][j] = result[1:]
                if not no_embedding_deeppavlov:
                    filtered_embeddings_list_deeppavlov.append(embeddings_list_deeppavlov[i])
                    no_embedding_deeppavlov_list = False
            if no_embedding_deeppavlov_list:
                no_embedding_deeppavlov_count += 1
        # Get LC-QuAD embeddings and template
        # Find embeddings
        embeddings_list = get_embeddings_schema_from_query(question['sparql_wikidata'])
        # Exclude questions without embeddings, because they can't be used for LSTM
        if not embeddings_list:
            no_embedding_count += 1
            no_embedding = True
        for i, element in enumerate(embeddings_list):
            if not no_embedding:
                result = find_element_DB(element, c)
                if result is None:
                    no_embedding_count += 1
                    no_embedding = True
                else:
                    embeddings_list[i] = result[1:]
        if not no_embedding:
            if not without_dp_embeddings:
                #question['deeppavlov_embeddings'] = embeddings_list_deeppavlov
                question['deeppavlov_embeddings'] = filtered_embeddings_list_deeppavlov
            question['embeddings'] = embeddings_list
            if first_element:
                # Append data to avoid memory error
                with open("data/LC_QuAD_2_" + filename + "_balanced_with_embeddings" + additional_name + ".json", "a") as set_file:
                    set_file.write("\n" + json.dumps(question, indent=2, ensure_ascii=False))
                first_element = False
            else:
                # Append data to avoid memory error
                with open("data/LC_QuAD_2_" + filename + "_balanced_with_embeddings" + additional_name + ".json", "a") as set_file:
                    set_file.write(",\n" + json.dumps(question, indent=2, ensure_ascii=False))
        if index % 100 == 0:
            print(index)
    c.close()
    if not without_dp_embeddings:
        #print("Questions without at least one DeepPavlov embedding: " + str(no_embedding_deeppavlov_count) + "/" + str(len(json_data)))
        print("Questions without DeepPavlov candidate queries: " + str(no_embedding_deeppavlov_count) + "/" + str(len(json_data)))
    print("Questions without at least one LC-QuAD 2.0 embedding " + str(no_embedding_count) + "/" + str(len(json_data)))
    # Append data to avoid memory error
    with open("data/LC_QuAD_2_" + filename + "_balanced_with_embeddings" + additional_name + ".json", "a") as set_file:
        set_file.write("\n]")

def count_answerable_and_unanswerable_questions(filename: str, additional_name: str = ""):
    with open("data/LC_QuAD_2_" + filename + "_balanced_with_embeddings" + additional_name + ".json", "r") as json_file:
        json_data = json.load(json_file)
    answerable_questions = 0
    unanswerable_questions = 0
    for element in json_data:
        if 'has_answer' in element and element['has_answer'] == False:
            answerable_questions += 1
        else:
            unanswerable_questions += 1
    print("Answerable questions: " + str(answerable_questions))
    print("Unanswerable questions: " + str(unanswerable_questions))
    print("Percentage answerable questions: " + str(answerable_questions * 100 / (answerable_questions + unanswerable_questions)))
    print("Percentage unanswerable questions: " + str(unanswerable_questions * 100 / (answerable_questions + unanswerable_questions)))

# Get a validation set from the training set, saving it in a new file. The validation size depends on a percentage of training examples given
# in input, and the selection is random using a seed
def get_validation_set(train_filename: str, valid_set_percent: float, seed: int):
    with open("data/" + train_filename + ".json", "r") as json_file:
        json_data = json.load(json_file)
    random.Random(seed).shuffle(json_data)
    valid_set_size = int(len(json_data) * valid_set_percent)
    valid_data = json_data[:valid_set_size]
    train_data = json_data[valid_set_size:]
    with open("data/" + train_filename + ".json", "w") as json_file:
        json.dump(train_data, json_file, indent=2, ensure_ascii=False)
    with open(("data/" + train_filename + ".json").replace("train", "valid", 1), "w") as json_file:
        json.dump(valid_data, json_file, indent=2, ensure_ascii=False)

# Add templates encodings, saving ground truth, model and converted ones. Converted are the ground truth encodings associated to DeepPavlov templates 
def save_encodings(filename: str, with_dp_embeddings: bool = True):
    with open("data/" + filename + ".json", "r") as json_file:
        json_data = json.load(json_file)
    for element in json_data:
        # Save the right template encoding
        subgraph_value = element['subgraph']
        if not subgraph_value:
            subgraph_value = "unknown"
        element['template_encoding'] = templates_encoding[subgraph_value][element['template_id']]
        if with_dp_embeddings:
            element['template_dp_encoding'] = templates_dp_encoding[element['deeppavlov_template']]
            # Save the right converted template encoding, using the model templates map to find the corresponding dataset template
            model_template = element['deeppavlov_template']
            subgraph = model_templates_map[model_template]['subgraph']
            template_id = model_templates_map[model_template]['template_id']
            element['template_conv_encoding'] = templates_encoding[subgraph][template_id]
    with open("data/" + filename + ".json", "w") as json_file:
        json.dump(json_data, json_file, indent=2, ensure_ascii=False)


# Prepare question data, including entities and relation lists
#preprocess_questions_file("train")
preprocess_questions_file("test")

#preprocess_questions_file("train", "_no_dp", True)

prepare_encodings()
#save_encodings("LC_QuAD_2_train_balanced_with_embeddings_no_dp", False)
#save_encodings("LC_QuAD_2_valid_balanced_with_embeddings_no_dp", False)
save_encodings("LC_QuAD_2_test_balanced_with_embeddings")

#get_validation_set("LC_QuAD_2_train_balanced_with_embeddings_no_dp", 0.15, 2021)

# Count answerable and unanswerable questions
#count_answerable_and_unanswerable_questions("train", "_no_dp")
#count_answerable_and_unanswerable_questions("valid", "_no_dp")
#count_answerable_and_unanswerable_questions("test")