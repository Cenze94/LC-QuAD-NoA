import questions_generator
import wikidata_ids_extractor

from typing import List, Dict, Any, Tuple
from copy import copy, deepcopy
import random
import re

# Most of required data for NNQT_question construction is common among all operations of the same template
def center_nnqt_question_construction(generated_template: Dict[str, Any]):
    questions_generator.recreate_nnqt_question(generated_template, "What is |property_0| of |entity_0|", [0], [0])

# Replace "James Madison" with a person entity, which cannot have any "addressee" property
def center_entity_example_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entity_id: str = "Q11813", old_entity: str = "James Madison") -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    questions_generator.entity_example_generation_common_part(current_uid, generated_template, generated_questions, old_entity_id, old_entity, \
        wikidata_ids_extractor.get_wikidata_person_types())
    center_nnqt_question_construction(generated_template)
    return generated_template

# Replace the known entity with a random candidate
def center_entity_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    questions_generator.entity_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    center_nnqt_question_construction(generated_template)
    return generated_template

# Replace "addressee" with another relation that doesn't link "James Madison" with any other entity
def center_relation_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [0], True)
    questions_generator.relation_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    center_nnqt_question_construction(generated_template)
    return generated_template

# Replace the known entity with another entity of the same instance or subclass type, that makes the query unable to return results
def center_entity_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    found, old_entities = questions_generator.entity_2_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    if not found:
        # There aren't valid candidates, so try with a random entity
        return center_entity_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        center_nnqt_question_construction(generated_template)
        return generated_template

# Replace the property with another property of the same instance or superproperty type, that makes the query unable to return results
def center_relation_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [0], True)
    found, old_properties = questions_generator.relation_2_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    if not found:
        # There aren't valid candidates, so try with a random property
        return center_relation_generation(current_uid, question_template, generated_questions, old_properties_ids, old_properties)
    else:
        center_nnqt_question_construction(generated_template)
        return generated_template

# Replace the object entity with another entity of the same instance or subclass type, taken from the other properties of the subject entity, so the new
# query becomes unable to return results
def center_entity_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    old_answer_id = next(iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values()))['value'].split("/")[-1]
    found, old_entities = questions_generator.entity_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, [
        "select ?ans ?ansLabel where {wd:" + old_answer_id + " ?rel ?ans . ?ans wdt:|rel_entity_type| wd:|entity_type| . ?ans rdfs:label ?ansLabel . " + \
            "FILTER (LANG(?ansLabel) = \"en\" && ?ans not in (wd:|old_entity_id|))} LIMIT 20"])
    if not found:
        # There aren't valid candidates, so try with a random entity of the same type or class
        return center_entity_2_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        center_nnqt_question_construction(generated_template)
        return generated_template

# Replace the property with another property that links the answer with another entity of the same type or class of the known entity
def center_relation_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [0], True)
    old_answer_id = next(iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values()))['value'].split("/")[-1]
    found, old_properties = questions_generator.relation_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_properties_ids, \
        ["select ?ans where {wd:" + old_answer_id + " ?ans ?obj . ?obj wdt:|rel_entity_type| wd:|entity_type| . FILTER (?ans not in (wdt:|old_property_id|))} LIMIT 20"])
    if not found:
        # There aren't valid candidates, so try with "relation_2" function
        return center_relation_2_generation(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    else:
        center_nnqt_question_construction(generated_template)
        return generated_template

# Most of required data for NNQT_question construction is common among all operations of the same template
def center_2_nnqt_question_construction(generated_template: Dict[str, Any]):
    questions_generator.recreate_nnqt_question(generated_template, "What is |property_0| of |entity_0|", [0], [0])

# Replace "United Nations Secretary-General" with a person entity, which cannot have any "nominated by" property
def center_2_entity_example_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entity_id: str = "Q81066", old_entity: str = "United Nations Secretary-General") -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    questions_generator.entity_example_generation_common_part(current_uid, generated_template, generated_questions, old_entity_id, old_entity, \
        wikidata_ids_extractor.get_wikidata_person_types())
    center_2_nnqt_question_construction(generated_template)
    return generated_template

# Replace the known entity with a random candidate
def center_2_entity_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    questions_generator.entity_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    center_2_nnqt_question_construction(generated_template)
    return generated_template

# Replace "nominated by" with another relation that doesn't link "United Nations Secretary-General" with any other entity
def center_2_relation_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [0], True)
    questions_generator.relation_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    center_2_nnqt_question_construction(generated_template)
    return generated_template

# Replace the known entity with another entity of the same instance or subclass type, that makes the query unable to return results
def center_2_entity_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    found, old_entities = questions_generator.entity_2_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    if not found:
        # There aren't valid candidates, so try with a random entity
        return center_2_entity_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        center_2_nnqt_question_construction(generated_template)
        return generated_template

# Replace the property with another property of the same instance or superproperty type, that makes the query unable to return results
def center_2_relation_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [0], True)
    found, old_properties = questions_generator.relation_2_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    if not found:
        # There aren't valid candidates, so try with a random property
        return center_2_relation_generation(current_uid, question_template, generated_questions, old_properties_ids, old_properties)
    else:
        center_2_nnqt_question_construction(generated_template)
        return generated_template

# Replace the subject entity with another entity of the same instance or subclass type, taken from the other properties of the object entity, so the new
# query becomes unable to return results. The same can be done taking the second object entity and the properties of the second subject. The choice between
# the two options, when both are feasible, is random
def center_2_entity_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    old_answer_id = next(iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values()))['value'].split("/")[-1]
    answer_var_name = re.findall(r'DISTINCT (\?\w*) WHERE', sparql_query, re.IGNORECASE)[0]
    entity_var_name = re.findall(r' (\?\w*)}', sparql_query)[0]
    modified_sparql_query = sparql_query.replace(answer_var_name, entity_var_name, 1)
    modified_answer_entity = questions_generator.get_sparql_query_results(modified_sparql_query)['results']['bindings'][0][entity_var_name[1:]]['value'].split("/")[-1]
    modified_answer_filter, _ = questions_generator.get_filter_from_element(modified_answer_entity, "obj", "", False)
    if modified_answer_filter:
        query = "select ?ans ?ansLabel where {?ans ?rel ?obj . ?ans wdt:|rel_entity_type| wd:|entity_type| . ?ans rdfs:label ?ansLabel . " + \
            "FILTER (" + modified_answer_filter + "LANG(?ansLabel) = \"en\" && ?ans not in (wd:|old_entity_id|))} LIMIT 20"
    else:
        query = "select ?ans ?ansLabel where {?ans ?rel wd:" + old_answer_id + " . ?ans wdt:|rel_entity_type| wd:|entity_type| . ?ans rdfs:label ?ansLabel . " + \
            "FILTER (LANG(?ansLabel) = \"en\" && ?ans not in (wd:|old_entity_id|))} LIMIT 20"
    found, old_entities = questions_generator.entity_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, [query])
    if not found:
        # There aren't valid candidates, so try with a random entity of the same type or class
        return center_2_entity_2_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        center_2_nnqt_question_construction(generated_template)
        return generated_template

# Replace the property with another property that links the answer with another entity of the same type or class of the known entity
def center_2_relation_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [0], True)
    old_answer_id = next(iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values()))['value'].split("/")[-1]
    answer_var_name = re.findall(r'DISTINCT (\?\w*) WHERE', sparql_query, re.IGNORECASE)[0]
    entity_var_name = re.findall(r' (\?\w*)}', sparql_query)[0]
    modified_sparql_query = sparql_query.replace(answer_var_name, entity_var_name, 1)
    modified_answer_entity = questions_generator.get_sparql_query_results(modified_sparql_query)['results']['bindings'][0][entity_var_name[1:]]['value'].split("/")[-1]
    modified_answer_filter, _ = questions_generator.get_filter_from_element(modified_answer_entity, "obj", "", False)
    if modified_answer_filter:
        query = "select ?ans where {?sbj ?ans ?obj . ?sbj wdt:|rel_entity_type| wd:|entity_type| . FILTER (" + modified_answer_filter + \
            "?ans not in (wdt:|old_property_id|) && REGEX(STR(?ans), \"P\\\\w+\"))} LIMIT 20"
    else:
        query = "select ?ans where {?sbj ?ans wd:" + old_answer_id + " . ?sbj wdt:|rel_entity_type| wd:|entity_type| . FILTER (?ans not in (wdt:|old_property_id|))} LIMIT 20"
    found, old_properties = questions_generator.relation_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_properties_ids, \
        [query])
    if not found:
        # There aren't valid candidates, so try with "relation_2" function
        return center_2_relation_2_generation(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    else:
        center_2_nnqt_question_construction(generated_template)
        return generated_template