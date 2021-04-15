import wikidata_ids_extractor

from typing import List, Dict, Any, Tuple, Union
from copy import copy, deepcopy
from SPARQLWrapper import SPARQLWrapper, JSON
from SPARQLWrapper.SPARQLExceptions import EndPointInternalError
from collections import OrderedDict
from urllib.error import HTTPError
from enum import Enum
from http.client import RemoteDisconnected
import requests
import random
import json
import re
import datetime
import time
import sys

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

# Check if a question has already been generated, and so should be deleted to avoid duplicates
def check_question_has_been_generated(sparql_query: str, generated_questions: List[Dict[str, Any]]) -> bool:
    for generated_question in generated_questions:
        if sparql_query == generated_question['sparql_wikidata']:
            return True
    # Check if the generated question has already been saved in JSON file
    with open("entities_and_properties/Generated_questions.json", "r") as json_file:
        json_data = json.load(json_file)
    old_generated_questions = json_data["generated_questions"]
    for generated_question in old_generated_questions:
        if sparql_query == generated_question['sparql_wikidata']:
            return True
    return False

# Get entity name using wbgetentities&ids Wikidata service. If entity is not an entity or a propriety, return an empty string
def get_entity_name_from_wikidata_id(entity_id: str) -> str:
    if len(re.findall(r'Q(\d+)', entity_id)) == 0 and len(re.findall(r'P(\d+)', entity_id)) == 0:
        return ""
    data = make_request('https://www.wikidata.org/w/api.php?action=wbgetentities&ids=' + entity_id + '&format=json').json()
    try:
        return data['entities'][entity_id]['labels']['en']['value'].replace("_", " ")
    except KeyError:
        return ""

# Execute SPARQL query and get results
def get_sparql_query_results(sparql_query: str) -> Dict[str, Any]:
    endpoint_url = "https://query.wikidata.org/sparql"
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    try:
        return sparql.query().convert()
    except HTTPError as e:
        # Manage "Too Many Requests" error
        timeout = get_delay(e.headers['retry-after'])
        print('Timeout {} m {} s'.format(timeout // 60, timeout % 60))
        time.sleep(timeout)
        return get_sparql_query_results(sparql_query)
    except EndPointInternalError as e:
        # Query requires too much time to execute, retry with a limit of 5 and if it not works return an empty result
        if "TimeoutException" in str(e):
            limit_number_list = re.findall(r'limit (\d+)', sparql_query, re.IGNORECASE)
            if len(limit_number_list) > 0:
                limit_number = limit_number_list[0]
            else:
                limit_number = ""
            if limit_number and int(limit_number) > 5:
                sparql_query = sparql_query.lower().replace("limit " + limit_number, "limit 5")
                print("Timeout! Retry with a limit of 5")
                return get_sparql_query_results(sparql_query)
            else:
                print("Timeout! Return an empty result")
                return {'results': {'bindings': []}}
        else:
            raise
    except RemoteDisconnected as e:
        print("Remote end closed connection without response")
        return {'results': {'bindings': []}}

# Check if the SPARQL query returns results
def check_sparql_query(sparql_query: str) -> bool:
    results = get_sparql_query_results(sparql_query)
    if 'count' in sparql_query.lower():
        # There is a COUNT in the query, so the empty answer is represented by a "0" value
        return next(iter(results['results']['bindings'][0].values()))['value'] != "0"
    else:
        return len(results['results']['bindings']) > 0

# Support function to get specific elements from SPARQL query, passing prefixes to identify them
def get_specific_elements_from_query(sparql_query: str, elements_positions: List[int], query_prefix: str, element_prefix: str) -> List[str]:
    elements_sparql = re.findall(query_prefix + ':(' + element_prefix + r'\d*)', sparql_query)
    elements_ids = []
    for element_index in elements_positions:
        if element_index >= len(elements_sparql):
            # There are less elements than expected, return the partial list
            break
        elements_ids.append(elements_sparql[element_index])
    return elements_ids

# Support function to get entities or properties from SPARQL query
def get_elements_from_query(sparql_query: str, elements_positions: List[int], is_property = False) -> List[str]:
    if is_property:
        return get_specific_elements_from_query(sparql_query, elements_positions, "wdt", "P")
    else:
        # I found cases of properties used as entities, manage them
        elements_entities = get_specific_elements_from_query(sparql_query, elements_positions, "wd", "Q")
        elements_properties = get_specific_elements_from_query(sparql_query, elements_positions, "wd", "P")
        if elements_properties:
            # There are properties used as entities, so we have to find the right order
            elements_prefixes = re.findall(r"wd:(\w)", sparql_query)
            elements = []
            elements_entities_iter = iter(elements_entities)
            elements_properties_iter = iter(elements_properties)
            for position, prefix in enumerate(elements_prefixes):
                if position in elements_positions:
                    if prefix == "Q":
                        elements.append(next(elements_entities_iter))
                    elif prefix == "P":
                        # Check to avoid strange behaviours
                        elements.append(next(elements_properties_iter))
            return elements
        else:
            # There aren't properties used as entities, return entities list
            return elements_entities

# Execute common operations, like uid and sparql query answers update
def update_template_with_common_data(generated_template: Dict[str, Any], current_uid: int):
    generated_template['uid'] = copy(current_uid)
    generated_template['has_answer'] = False
    generated_template['sparql_dbpedia18'] = ""

class ElementType(Enum):
    entity = "entity" # Useless value
    relation = "relation" # Useless value
    number = "REGEX(STR(|var|), \"^\\\\d+$\")"
    date = "REGEX(STR(|var|), \"^(\\\\d+)-(\\\\d+)-(\\\\d+)T\")"
    string = "!REGEX(STR(|var|), \"^\\\\d+$\") && !REGEX(STR(|var|), \"^(\\\\d+)-(\\\\d+)-(\\\\d+)T\") && !REGEX(STR(|var|), \"^Q(\\\\d+)\")"

# Get type of a non entity element
def get_element_type(element: str) -> ElementType:
    if len(re.findall(r'Q(\d+)', element)) > 0:
        return ElementType.entity
    if len(re.findall(r'P(\d+)', element)) > 0:
        return ElementType.property
    if len(re.findall(r'^(\d+$)', element)) > 0:
        return ElementType.number
    if len(re.findall(r'^(\d+)-(\d+)-(\d+)T', element)) > 0:
        return ElementType.date
    return ElementType.string

# Support function, check element type and return the corresponding filter
def get_filter_from_element(element: str, var_name: str, prop_statement_name: str, add_statement_filter: bool = True) -> str:
    element_type = get_element_type(element)
    if element_type == ElementType.number:
        query_filter =  r"REGEX(STR(?" + var_name + "), \"^\\\\d+$\") && "
    elif element_type == ElementType.date:
        query_filter = r"REGEX(STR(?" + var_name + "), \"^(\\\\d+)-(\\\\d+)-(\\\\d+)T\") && "
    elif element_type == ElementType.string:
        # Element is a string
        query_filter = r"!REGEX(STR(?" + var_name + "), \"^\\\\d+$\") && !REGEX(STR(?" + var_name + "), \"^(\\\\d+)-\") && !REGEX(STR(?" + var_name + "), \"^(\\\\d+)-(\\\\d+)-(\\\\d+)T\") && "
    else:
        # Element is a property or an entity
        query_filter = ""
    if add_statement_filter:
        # Add property statement format check
        query_filter += r"REGEX(STR(?" + prop_statement_name + "), \"Q(\\\\d+)-\")"
    return query_filter, element_type

# Create the NNQT_question using a template specific text, so the updated entity or property label will surely be present. If there are elements that are not entities,
# they must be specified into "fixed_entities"
def recreate_nnqt_question(generated_template: Dict[str, Any], template_text: str, entities_indexes: List[int], properties_indexes: List[int], wdt_prefixes: bool = True,
fixed_entities: List[str] = None):
    sparql_query = generated_template['sparql_wikidata']
    entities_ids = get_elements_from_query(sparql_query, entities_indexes)
    if wdt_prefixes:
        properties_ids = get_elements_from_query(sparql_query, properties_indexes, True)
    else:
        # Properties refers to qualifiers, so there is a "p:" prefix and the remaining elements are "pq:". In this case "properties_indexes" contains "pq" properties indexes
        properties_ids = get_specific_elements_from_query(sparql_query, [0], "p", "P")
        properties_ids.extend(get_specific_elements_from_query(sparql_query, properties_indexes, "pq", "P"))
    for index, entity_id in enumerate(entities_ids):
        entity = get_entity_name_from_wikidata_id(entity_id)
        template_text = template_text.replace("|entity_" + str(index) + "|", "{" + entity + "}")
    if fixed_entities:
        for index, fixed_entity in enumerate(fixed_entities):
            template_text = template_text.replace("|element_" + str(index) + "|", "{" + fixed_entity + "}")
    for index, property_id in enumerate(properties_ids):
        property_name = get_entity_name_from_wikidata_id(property_id)
        template_text = template_text.replace("|property_" + str(index) + "|", "{" + property_name + "}")
    generated_template['NNQT_question'] = template_text

# Support function to get possible entities with a specific property and linked to the answer (so for entity_3 functions)
def get_possible_entities_from_entity_type(old_entity_id: str, ent_property: str, sparql_query: str) -> Tuple[List[str], List[str]]:
    old_entity_types = get_sparql_query_results("select distinct ?obj where {wd:" + old_entity_id + " wdt:" + ent_property + " ?obj}")['results']['bindings']
    sparql_query = sparql_query.replace("|old_entity_id|", old_entity_id).replace("|rel_entity_type|", ent_property)
    new_possible_entities_ids = []
    new_possible_entities = []
    for old_entity_type in old_entity_types:
        old_entity_type_extracted = old_entity_type['obj']['value'].split("/")[-1]
        current_sparql_query = sparql_query.replace("|entity_type|", old_entity_type_extracted)
        # Get all the entities of the same type of the old entity and different from it
        new_possible_entities_results = get_sparql_query_results(current_sparql_query)['results']['bindings']
        for possible_entity in new_possible_entities_results:
            possible_entity_id = possible_entity['ans']['value'].split("/")[-1]
            # If entity has already been found, for example because it has two instance or subclass in common with the original entity, skip it
            if possible_entity_id not in new_possible_entities_ids:
                new_possible_entities_ids.append(possible_entity_id)
                new_possible_entities.append(possible_entity['ansLabel']['value'].split("/")[-1])
    return new_possible_entities_ids, new_possible_entities

# Some questions use numbers and texts as object entities, so they have to be treaty differently because of the lack of a type or subclass. They are
# considered two separated classes
def get_possible_properties_from_non_entity(old_entity_id: str, sparql_query: str, old_property_id: str, old_property_prefix: str = None) -> Tuple[List[str], List[str]]:
    new_possible_properties_ids = []
    new_possible_properties = []
    element_type = get_element_type(old_entity_id)
    if element_type != ElementType.entity:
        # Element is not an entity, replace filter
        filter_var_name = re.findall("\|filter\|(\?\w+)\|", sparql_query)[0]
        sparql_query = sparql_query.replace("|filter|" + filter_var_name + "|", element_type.value.replace("|var|", filter_var_name) + " && REGEX(STR(?ans), \"P\\\\w+\")")\
            .replace("|old_property_id|", old_property_id)
        new_possible_properties_results = get_sparql_query_results(sparql_query)['results']['bindings']
        # Add distinct properties that are linked to the entity through the right property prefix
        for possible_property in new_possible_properties_results[:]:
            if old_property_prefix:
                print(possible_property)
            possible_property_id = possible_property['ans']['value'].split("/")[-1]
            # Skip property if it has already been found
            if possible_property_id not in new_possible_properties_ids:
                new_possible_properties_ids.append(possible_property_id)
                new_possible_properties.append(get_entity_name_from_wikidata_id(possible_property_id))
    return new_possible_properties_ids, new_possible_properties

# Support function similar to the above one, that returns the list of properties IDs and names instead of entities one (so for relation_3 functions)
def get_possible_properties_from_entity_type(old_entity_id: str, ent_property: str, sparql_query: str, old_property_id: str, \
    old_property_prefix: str = None) -> Tuple[List[str], List[str]]:
    # Check if entity is a number or a text string, if so return the resulting output
    new_possible_properties_ids, new_possible_properties = get_possible_properties_from_non_entity(old_entity_id, sparql_query, old_property_id, old_property_prefix)
    if new_possible_properties_ids:
        return new_possible_properties_ids, new_possible_properties
    # Entity is not a number or a text string
    old_entity_types = get_sparql_query_results("select distinct ?obj where {wd:" + old_entity_id + " wdt:" + ent_property + " ?obj}")['results']['bindings']
    sparql_query = sparql_query.replace("|old_entity_id|", old_entity_id).replace("|rel_entity_type|", ent_property).replace("|old_property_id|", old_property_id)
    for old_entity_type in old_entity_types:
        old_entity_type_extracted = old_entity_type['obj']['value'].split("/")[-1]
        current_sparql_query = sparql_query.replace("|entity_type|", old_entity_type_extracted)
        # Get all the properties linked to entities of the same type of the old entity and different from it
        new_possible_properties_results = get_sparql_query_results(current_sparql_query)['results']['bindings']
        for possible_property in new_possible_properties_results:
            possible_property_id = possible_property['ans']['value'].split("/")[-1]
            # Skip property if it has already been found
            if possible_property_id not in new_possible_properties_ids and not old_property_prefix:
                new_possible_properties_ids.append(possible_property_id)
                new_possible_properties.append(get_entity_name_from_wikidata_id(possible_property_id))
            elif possible_property_id not in new_possible_properties_ids:
                # If "old_property_prefix" is not None verify if property address contains the corresponding prefix
                if (old_property_prefix == "pq" and "qualifier" in possible_property['ans']['value']) or \
                    (old_property_prefix == "ps" and "statement" in possible_property['ans']['value']):
                    new_possible_properties_ids.append(possible_property_id)
                    new_possible_properties.append(get_entity_name_from_wikidata_id(possible_property_id))
    return new_possible_properties_ids, new_possible_properties

# Support function to get possible entities with a specific property (so for entity_2 functions)
def get_possible_entities_from_property(ent_property: str, old_entity_id: str) -> Tuple[List[str], List[str]]:
    return get_possible_entities_from_entity_type(old_entity_id, ent_property, "select ?ans ?ansLabel where {?ans wdt:|rel_entity_type| wd:|entity_type| ; rdfs:label " + \
        "?ansLabel . FILTER (LANG(?ansLabel) = \"en\" && ?ans not in (wd:|old_entity_id|))} LIMIT 20")

# Support function, try to replace a given entity with every possible candidate from a list, until we found one that doesn't return results or have tried
# with all candidates. The candidate choice is done randomly to improve the variety of solutions, for example if the same template is used for multiple questions
def try_possible_entities(generated_template: Dict[str, Any], generated_questions: List[Dict[str, Any]], new_possible_entities_ids: List[str],
new_possible_entities: List[str], old_entity_id: str, old_entity: str) -> bool:
    while len(new_possible_entities) > 0:
        index = random.randrange(len(new_possible_entities))
        new_entity_id = new_possible_entities_ids.pop(index)
        new_entity = new_possible_entities.pop(index)
        # old_entity_id value might be contained in another property, that shouldn't be replaced, for example "P19" and "P190". To avoid this behaviour try with every possible
        # character that can be found after the property
        sparql_query = copy(generated_template['sparql_wikidata']).replace(old_entity_id + " ", new_entity_id + " ").replace(old_entity_id + ".", new_entity_id + ".") \
        .replace(old_entity_id + "}", new_entity_id + "}")
        if not check_sparql_query(sparql_query) and not check_question_has_been_generated(sparql_query, generated_questions):
            generated_template['NNQT_question'] = generated_template['NNQT_question'].replace(old_entity, new_entity)
            if generated_template['question']:
                generated_template['question'] = generated_template['question'].replace(old_entity, new_entity)
            if generated_template['paraphrased_question']:
                generated_template['paraphrased_question'] = generated_template['paraphrased_question'].replace(old_entity, new_entity)
            generated_template['sparql_wikidata'] = sparql_query
            return True
    return False

# Get all possible entities candidates for every entity given in input, excluding ones without possible entities. Since I can do the same for properties changing
# only one line, I adapted the function to this case
def get_possible_entities_lists(old_entities_ids_input: List[str], old_entities_input: List[str], is_property: bool = False) -> Tuple[List[List[str]], List[List[str]], List[str], List[str]]:
    old_entities_ids = copy(old_entities_ids_input)
    old_entities = copy(old_entities_input)
    new_possible_entities_ids_list = []
    new_possible_entities_list = []
    for old_entity_id in old_entities_ids[:]:
        new_possible_entities_ids, new_possible_entities = get_possible_entities_from_property("P31", old_entity_id)
        if is_property:
            temp_list_ids, temp_list = get_possible_entities_from_property("P1647", old_entity_id)
        else:
            temp_list_ids, temp_list = get_possible_entities_from_property("P279", old_entity_id)
        for i, temp_id in enumerate(temp_list_ids):
            # Add entity to list only if has not already been added
            if not temp_id in new_possible_entities_ids:
                new_possible_entities_ids.append(temp_id)
                new_possible_entities.append(temp_list[i])
        if new_possible_entities_ids:
            new_possible_entities_ids_list.append(new_possible_entities_ids)
            new_possible_entities_list.append(new_possible_entities)
        else:
            index = old_entities_ids.index(old_entity_id)
            del old_entities_ids[index]
            del old_entities[index]
    return new_possible_entities_ids_list, new_possible_entities_list, old_entities_ids, old_entities

# Get all possible entities candidates for every entity given in input, excluding ones without possible entitities. A possible entity is an entity different from
# the original one and linked with the answer. Each old_entities_data element has an entity ID as key and is a dictionary with 2 values: the corresponding name and
# the corresponding sparql query. In the query must be present 2 keywords, i. e. "|old_entity_id|", which indicates where the analysed entity ID has to be placed,
# and "|entity_type|", that signals where the common entity type is needed
def get_possible_entities_from_entity_lists(old_entities_data: Dict[str, Dict[str, str]]) -> Tuple[List[List[str]], List[List[str]], List[str], List[str]]:
    new_possible_entities_ids_list = []
    new_possible_entities_list = []
    for key in list(old_entities_data):
        old_entity_id = old_entities_data[key]['id']
        sparql_query = old_entities_data[key]['sparql_query']
        # Distinguish between instance of and subclass of types, to avoid results of wrong type (e. g. an entity instance of another entity that is a superclass of
        # the original entity)
        new_possible_entities_ids, new_possible_entities = get_possible_entities_from_entity_type(old_entity_id, "P31", sparql_query)
        temp_list_ids, temp_list = get_possible_entities_from_entity_type(old_entity_id, "P279", sparql_query)
        for i, temp_id in enumerate(temp_list_ids):
            # Add entity to list only if has not already been added
            if not temp_id in new_possible_entities_ids:
                new_possible_entities_ids.append(temp_id)
                new_possible_entities.append(temp_list[i])
        # If there aren't candidate entities associated to this entity, ignore it
        if new_possible_entities_ids:
            new_possible_entities_ids_list.append(new_possible_entities_ids)
            new_possible_entities_list.append(new_possible_entities)
        else:
            del old_entities_data[key]
    old_entities_ids = []
    old_entities = []
    for old_entity_element in old_entities_data.values():
        old_entities_ids.append(old_entity_element['id'])
        old_entities.append(old_entity_element['name'])
    return new_possible_entities_ids_list, new_possible_entities_list, old_entities_ids, old_entities

# Support function, very similar to the above one, that returns properties data. In the query there is one more keyword: "|old_property_id|", that indicates where the
# analysed property ID has to be placed. "old_properties_prefixes" contains the alternative prefixes for properties if they are not all "wdt"
def get_possible_properties_from_entity_lists(old_entities_data: Dict[str, Dict[str, str]], old_properties_ids: List[str], old_properties: List[str], \
    old_properties_prefixes: List[str] = None) -> Tuple[List[List[str]], List[List[str]], List[str], List[str]]:
    new_possible_properties_ids_list = []
    new_possible_properties_list = []
    for key in list(old_entities_data):
        old_entity_id = old_entities_data[key]['id']
        old_property_id = old_entities_data[key]['linked_property_id']
        old_property = old_entities_data[key]['linked_property']
        if 'linked_property_prefix' in old_entities_data[key]:
            old_property_prefix = old_entities_data[key]['linked_property_prefix']
        else:
            old_property_prefix = None
        sparql_query = old_entities_data[key]['sparql_query']
        # Distinguish between instance of and subclass of types, to avoid results of wrong type (e. g. an entity instance of another entity that is a superclass of
        # the original entity)
        new_possible_properties_ids, new_possible_properties = get_possible_properties_from_entity_type(old_entity_id, "P31", sparql_query, old_property_id, old_property_prefix)
        temp_list_ids, temp_list = get_possible_properties_from_entity_type(old_entity_id, "P279", sparql_query, old_property_id, old_property_prefix)
        for i, temp_id in enumerate(temp_list_ids):
            # Add property to list only if has not already been added
            if not temp_id in new_possible_properties_ids:
                new_possible_properties_ids.append(temp_id)
                new_possible_properties.append(temp_list[i])
        # If there aren't candidate properties associated to this property, ignore it
        if new_possible_properties_ids:
            new_possible_properties_ids_list.append(new_possible_properties_ids)
            new_possible_properties_list.append(new_possible_properties)
        else:
            del old_entities_data[key]
    old_properties_ids = []
    old_properties = []
    for old_entity_element in old_entities_data.values():
        old_properties_ids.append(old_entity_element['linked_property_id'])
        old_properties.append(old_entity_element['linked_property'])
    return new_possible_properties_ids_list, new_possible_properties_list, old_properties_ids, old_properties

# Since in this type of operation the only difference between templates is the extraction and selection of the entity to replace, define a common part to reduce code repetition
def entity_example_generation_common_part(current_uid: int, generated_template: Dict[str, Any], generated_questions: List[Dict[str, Any]], old_entity_id: str, old_entity: str,
entities_source: List[wikidata_ids_extractor.DataType]):
    # Save original sparql wikidata for debugging if not present
    if not 'old_sparql_wikidata' in generated_template.keys():
        generated_template['old_sparql_wikidata'] = copy(generated_template['sparql_wikidata'])
    # If the chosen random entity returns results a new random entity ID is needed
    while True:
        new_entity_id, new_entity = wikidata_ids_extractor.get_random_wikidata_entity_from_list(entities_source)
        sparql_query = copy(generated_template['sparql_wikidata']).replace(old_entity_id, new_entity_id)
        if not check_sparql_query(sparql_query) and not check_question_has_been_generated(sparql_query, generated_questions):
            generated_template['NNQT_question'] = generated_template['NNQT_question'].replace(old_entity, new_entity)
            if generated_template['question']:
                generated_template['question'] = generated_template['question'].replace(old_entity, new_entity)
            if generated_template['paraphrased_question']:
                generated_template['paraphrased_question'] = generated_template['paraphrased_question'].replace(old_entity, new_entity)
            generated_template['sparql_wikidata'] = sparql_query
            break
    generated_template['operation_type'] = "entity"
    generated_template['operation_difficulty'] = 1
    update_template_with_common_data(generated_template, current_uid)

# Since in this type of operation the only difference between templates is the extraction and selection of the entity to replace, define a common part to reduce code repetition
def entity_generation_common_part(current_uid: int, generated_template: Dict[str, Any], generated_questions: List[Dict[str, Any]], old_entities_ids: List[str],
old_entities: List[str] = None):
    if not old_entities:
        # Get old entities names
        old_entities = []
        for old_entity_id in old_entities_ids:
            old_entities.append(get_entity_name_from_wikidata_id(old_entity_id).replace("_", " "))
    # Try replacing a random entity with a random candidate
    index = random.randrange(len(old_entities_ids))
    old_entity_id = old_entities_ids[index]
    old_entity = old_entities[index]
    # Save original sparql wikidata for debugging if not present
    if not 'old_sparql_wikidata' in generated_template.keys():
        generated_template['old_sparql_wikidata'] = copy(generated_template['sparql_wikidata'])
    # If the chosen random entity returns results we need a new random entity ID
    while True:
        new_entity_id, new_entity = wikidata_ids_extractor.get_random_wikidata_entity_from_all()
        sparql_query = copy(generated_template['sparql_wikidata']).replace(old_entity_id, new_entity_id)
        if not check_sparql_query(sparql_query) and not check_question_has_been_generated(sparql_query, generated_questions):
            generated_template['NNQT_question'] = generated_template['NNQT_question'].replace(old_entity, new_entity)
            if generated_template['question']:
                generated_template['question'] = generated_template['question'].replace(old_entity, new_entity)
            if generated_template['paraphrased_question']:
                generated_template['paraphrased_question'] = generated_template['paraphrased_question'].replace(old_entity, new_entity)
            generated_template['sparql_wikidata'] = sparql_query
            break
    generated_template['operation_type'] = "entity"
    generated_template['operation_difficulty'] = 1
    update_template_with_common_data(generated_template, current_uid)

# Since in this type of operation the only difference between templates is the extraction and selection of the relation to replace, define a common part to reduce code repetition
def relation_generation_common_part(current_uid: int, generated_template: Dict[str, Any], generated_questions: List[Dict[str, Any]], old_properties_ids: List[str],
old_properties: List[str] = None):
    if not old_properties:
        # Get old properties names
        old_properties = []
        for old_property_id in old_properties_ids:
            old_properties.append(get_entity_name_from_wikidata_id(old_property_id).replace("_", " "))
    # Select a random relation to replace
    index = random.randrange(len(old_properties))
    old_property_id = old_properties_ids[index]
    old_property = old_properties[index]
    # Save original sparql wikidata for debugging if not present
    if not 'old_sparql_wikidata' in generated_template.keys():
        generated_template['old_sparql_wikidata'] = copy(generated_template['sparql_wikidata'])
    # If the chosen random property returns results we need a new random property ID
    while True:
        new_property_id, new_property = wikidata_ids_extractor.get_random_wikidata_property()
        sparql_query = copy(generated_template['sparql_wikidata']).replace(old_property_id, new_property_id)
        if not check_sparql_query(sparql_query) and not check_question_has_been_generated(sparql_query, generated_questions):
            generated_template['NNQT_question'] = generated_template['NNQT_question'].replace(old_property, new_property)
            if generated_template['question']:
                generated_template['question'] = generated_template['question'].replace(old_property, new_property)
            if generated_template['paraphrased_question']:
                generated_template['paraphrased_question'] = generated_template['paraphrased_question'].replace(old_property, new_property)
            generated_template['sparql_wikidata'] = sparql_query
            break
    generated_template['operation_type'] = "relation"
    generated_template['operation_difficulty'] = 1
    update_template_with_common_data(generated_template, current_uid)

# Since in these general functions the difference between every template should be only entities extraction, define a common part to reduce code repetition
def entity_2_generation_common_part(current_uid: int, generated_template: Dict[str, Any], generated_questions: List[Dict[str, Any]], old_entities_ids: List[str],
old_entities: List[str] = None) -> Tuple[bool, List[str]]:
    # Save original sparql wikidata for debugging if not present
    if not 'old_sparql_wikidata' in generated_template.keys():
        generated_template['old_sparql_wikidata'] = copy(generated_template['sparql_wikidata'])
    if not old_entities:
        # Get old entities names
        old_entities = []
        for old_entity_id in old_entities_ids:
            old_entities.append(get_entity_name_from_wikidata_id(old_entity_id).replace("_", " "))
    # Get filtered entities to give priority to ones with new possible entities
    new_possible_entities_ids_list, new_possible_entities_list, old_entities_ids_filtered, old_entities_filtered = get_possible_entities_lists(old_entities_ids, old_entities)
    found = False
    while len(old_entities_filtered) > 0 and not found:
        # The old entity choice is done randomly to improve the variety of solutions
        index = random.randrange(len(old_entities_filtered))
        new_possible_entities_ids = new_possible_entities_ids_list.pop(index)
        new_possible_entities = new_possible_entities_list.pop(index)
        old_entity_id = old_entities_ids_filtered.pop(index)
        old_entity = old_entities_filtered.pop(index)
        # Try every possible entity until we found one that doesn't return any result
        found = try_possible_entities(generated_template, generated_questions, new_possible_entities_ids, new_possible_entities, old_entity_id, old_entity)
    if found:
        generated_template['operation_type'] = "entity"
        generated_template['operation_difficulty'] = 2
        update_template_with_common_data(generated_template, current_uid)
    return found, old_entities

# Since these functions are almost identical, define a common part to reduce code repetition
def relation_2_generation_common_part(current_uid: int, generated_template: Dict[str, Any], generated_questions: List[Dict[str, Any]], old_properties_ids: List[str],
old_properties: List[str] = None) -> Tuple[bool, List[str]]:
    # Save original sparql wikidata for debugging if not present
    if not 'old_sparql_wikidata' in generated_template.keys():
        generated_template['old_sparql_wikidata'] = copy(generated_template['sparql_wikidata'])
    if not old_properties:
        # Get old properties names
        old_properties = []
        for old_property_id in old_properties_ids:
            old_properties.append(get_entity_name_from_wikidata_id(old_property_id).replace("_", " "))
    # Get filtered properties to give priority to ones with new possible properties, using the properties version of "get_possible_entities_lists"
    new_possible_properties_ids_list, new_possible_properties_list, old_properties_ids_filtered, old_properties_filtered = \
        get_possible_entities_lists(old_properties_ids, old_properties, True)
    found = False
    while len(old_properties_filtered) > 0 and not found:
        # The old property choice is done randomly to improve the variety of solutions
        index = random.randrange(len(old_properties_filtered))
        new_possible_properties_ids = new_possible_properties_ids_list.pop(index)
        new_possible_properties = new_possible_properties_list.pop(index)
        old_property_id = old_properties_ids_filtered.pop(index)
        old_property = old_properties_filtered.pop(index)
        # Try every possible property until we found one that doesn't return any result. Despite being built for entities, this function works also for properties
        found = try_possible_entities(generated_template, generated_questions, new_possible_properties_ids, new_possible_properties, old_property_id, old_property)
    if found:
        generated_template['operation_type'] = "relation"
        generated_template['operation_difficulty'] = 2
        update_template_with_common_data(generated_template, current_uid)
    return found, old_properties

# Since these functions are almost identical, define a common part to reduce code repetition
def entity_3_generation_common_part(current_uid: int, generated_template: Dict[str, Any], generated_questions: List[Dict[str, Any]], old_entities_ids: List[str], \
    old_entities_queries: List[str]) -> Tuple[bool, List[str]]:
    # Save original sparql wikidata for debugging if not present
    if not 'old_sparql_wikidata' in generated_template.keys():
        generated_template['old_sparql_wikidata'] = copy(generated_template['sparql_wikidata'])
    # Get old entities names and prepare dictionary
    old_entities = []
    old_entity_data = OrderedDict()
    for index, old_entity_id in enumerate(old_entities_ids):
        old_entity = get_entity_name_from_wikidata_id(old_entity_id).replace("_", " ")
        old_entities.append(old_entity)
        old_entity_data[index] = {
            "id": old_entity_id,
            "name": old_entity,
            "sparql_query": old_entities_queries[index]
        }
    new_possible_entities_ids_list, new_possible_entities_list, old_entities_ids_filtered, old_entities_filtered = get_possible_entities_from_entity_lists(old_entity_data)
    found = False
    while len(old_entities_ids_filtered) > 0 and not found:
        # The old entity choice is done randomly to improve the variety of solutions
        index = random.randrange(len(old_entities_ids_filtered))
        new_possible_entities_ids = new_possible_entities_ids_list.pop(index)
        new_possible_entities = new_possible_entities_list.pop(index)
        old_entity_id = old_entities_ids_filtered.pop(index)
        old_entity = old_entities_filtered.pop(index)
        # Try every possible entity until we found one that doesn't return any result
        found = try_possible_entities(generated_template, generated_questions, new_possible_entities_ids, new_possible_entities, old_entity_id, old_entity)
    if found:
        generated_template['operation_type'] = "entity"
        generated_template['operation_difficulty'] = 3
        update_template_with_common_data(generated_template, current_uid)
    return found, old_entities

# These functions are almost identical, so define a common part to reduce code repetition. "old_entities_ids" is the list of entities associated to properties, since every
# property candidate depends on entities candidate of the associated entity
def relation_3_generation_common_part(current_uid: int, generated_template: Dict[str, Any], generated_questions: List[Dict[str, Any]], old_entities_ids: List[str], \
    old_properties_ids: List[str], old_entities_queries: List[str], old_properties_prefixes: List[str] = None) -> Tuple[bool, List[str]]:
    # Save original sparql wikidata for debugging if not present
    if not 'old_sparql_wikidata' in generated_template.keys():
        generated_template['old_sparql_wikidata'] = copy(generated_template['sparql_wikidata'])
    # Get old properties names
    old_properties = []
    for old_property_id in old_properties_ids:
        old_properties.append(get_entity_name_from_wikidata_id(old_property_id))
    # Get old entities names and prepare dictionary
    old_entities = []
    old_entity_data = OrderedDict()
    for index, old_entity_id in enumerate(old_entities_ids):
        old_entity = get_entity_name_from_wikidata_id(old_entity_id).replace("_", " ")
        old_entities.append(old_entity)
        old_entity_data[index] = {
            "id": old_entity_id,
            "name": old_entity,
            "sparql_query": old_entities_queries[index],
            "linked_property_id": old_properties_ids[index],
            "linked_property": old_properties[index]
        }
        if old_properties_prefixes:
            old_entity_data[index]["linked_property_prefix"] = old_properties_prefixes[index]
    new_possible_properties_ids_list, new_possible_properties_list, old_properties_ids_filtered, old_properties_filtered, = get_possible_properties_from_entity_lists(
        old_entity_data, old_properties_ids, old_properties, old_properties_prefixes)
    found = False
    while len(old_properties_ids_filtered) > 0 and not found:
        # The old entity choice is done randomly to improve the variety of solutions
        index = random.randrange(len(old_properties_ids_filtered))
        new_possible_properties_ids = new_possible_properties_ids_list.pop(index)
        new_possible_properties = new_possible_properties_list.pop(index)
        old_property_id = old_properties_ids_filtered.pop(index)
        old_property = old_properties_filtered.pop(index)
        # Try every possible property until we found one that doesn't return any result. Despite being built for entities, this function works also for properties
        found = try_possible_entities(generated_template, generated_questions, new_possible_properties_ids, new_possible_properties, old_property_id, old_property)
    if found:
        generated_template['operation_type'] = "relation"
        generated_template['operation_difficulty'] = 3
        update_template_with_common_data(generated_template, current_uid)
    return found, old_properties
