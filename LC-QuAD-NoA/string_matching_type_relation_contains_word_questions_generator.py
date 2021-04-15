import questions_generator
import wikidata_ids_extractor

from typing import List, Dict, Any, Tuple
from copy import copy, deepcopy
import re

# Most of required data for NNQT_question construction is common among all operations of the same template
def string_matching_type_relation_contains_word_nnqt_question_construction(generated_template: Dict[str, Any]):
    # There are 2 templates covered in this file, which have different NNQT_question construction. The extraction of word or letter is common
    word = re.findall("'(.*?)'", generated_template['sparql_wikidata'])[0]
    if generated_template['template_id'] == 3:
        # Word matching
        questions_generator.recreate_nnqt_question(generated_template, "Give me |entity_0| that |property_0| |entity_1| and which contains the word {" + word + "} in their name", [0, 1], [1])
    else:
        # Starting letter matching
        questions_generator.recreate_nnqt_question(generated_template, "Give me |entity_0| that |property_0| |entity_1| and which that starts with {'" + word + "'}", [0, 1], [1])

# Replace "historical period" with a politician or leader entity that isn't the type of an entity linked with "Muromachi period" through "time period" property
def string_matching_type_relation_contains_word_entity_example_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entity_id: str = "Q11514315", old_entity: str = "historical period") -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    questions_generator.entity_example_generation_common_part(current_uid, generated_template, generated_questions, old_entity_id, old_entity, \
        [wikidata_ids_extractor.DataType.politicians_and_leaders])
    string_matching_type_relation_contains_word_nnqt_question_construction(generated_template)
    return generated_template

# Replace the known entity with a random candidate
def string_matching_type_relation_contains_word_entity_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    questions_generator.entity_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    string_matching_type_relation_contains_word_nnqt_question_construction(generated_template)
    return generated_template

# Replace "time period" property, without considering "instance of" because it's fixed and is not the only relation of this template
def string_matching_type_relation_contains_word_relation_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [1], True)
    questions_generator.relation_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    string_matching_type_relation_contains_word_nnqt_question_construction(generated_template)
    return generated_template

# Replace one of the known entities with another entity of the same instance or subclass type, that makes the query unable to return results
def string_matching_type_relation_contains_word_entity_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    found, old_entities = questions_generator.entity_2_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    if not found:
        # There aren't valid candidates, so try with a random entity
        return string_matching_type_relation_contains_word_entity_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        string_matching_type_relation_contains_word_nnqt_question_construction(generated_template)
        return generated_template

# Replace the known property with another property of the same instance or superproperty type, that makes the query unable to return results
def string_matching_type_relation_contains_word_relation_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [1], True)
    found, old_properties = questions_generator.relation_2_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    if not found:
        # There aren't valid candidates, so try with a random property
        return string_matching_type_relation_contains_word_relation_generation(current_uid, question_template, generated_questions, old_properties_ids, old_properties)
    else:
        string_matching_type_relation_contains_word_nnqt_question_construction(generated_template)
        return generated_template

# Replace the object entity with another entity of the same instance or subclass type, taken from the other properties of the subject entity, so the new
# query becomes unable to return results. The same can be done taking the second object entity and the properties of the second subject. The choice between
# the two options, when both are feasible, is random
def string_matching_type_relation_contains_word_entity_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    old_answer_id = next(iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values()))['value'].split("/")[-1]
    found, old_entities = questions_generator.entity_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, [
        "select ?ans ?ansLabel where {?ans ?rel wd:" + old_answer_id + " . ?ans wdt:|rel_entity_type| wd:|entity_type| . ?ans rdfs:label ?ansLabel . " + \
            "FILTER (LANG(?ansLabel) = \"en\" && ?ans not in (wd:|old_entity_id|))} LIMIT 20",
        "select ?ans ?ansLabel where {?ans ?rel wd:" + old_answer_id + " . ?ans wdt:|rel_entity_type| wd:|entity_type| . ?ans rdfs:label ?ansLabel . " + \
            "FILTER (LANG(?ansLabel) = \"en\" && ?ans not in (wd:|old_entity_id|))} LIMIT 20"])
    if not found:
        # There aren't valid candidates, so try with a random entity of the same type or class
        return string_matching_type_relation_contains_word_entity_2_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        string_matching_type_relation_contains_word_nnqt_question_construction(generated_template)
        return generated_template

# Replace the known property with another property that links the answer with another entity of the same type or class of the second known entity
def string_matching_type_relation_contains_word_relation_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [1])
    old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [1], True)
    old_answer_id = next(iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values()))['value'].split("/")[-1]
    found, old_properties = questions_generator.relation_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_properties_ids, \
        ["select ?ans where {?sbj ?ans wd:" + old_answer_id + " . ?sbj wdt:|rel_entity_type| wd:|entity_type| . FILTER (?ans not in (wdt:|old_property_id|))} LIMIT 20"])
    if not found:
        # There aren't valid candidates, so try with "relation_2" function
        return string_matching_type_relation_contains_word_relation_2_generation(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    else:
        string_matching_type_relation_contains_word_nnqt_question_construction(generated_template)
        return generated_template