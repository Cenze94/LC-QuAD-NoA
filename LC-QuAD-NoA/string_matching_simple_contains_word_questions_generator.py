import questions_generator
import wikidata_ids_extractor

from typing import List, Dict, Any, Tuple
from copy import copy, deepcopy
import re

# Most of required data for NNQT_question construction is common among all operations of the same template
def string_matching_simple_contains_word_nnqt_question_construction(generated_template: Dict[str, Any]):
    # There are 2 templates covered in this file, which have different NNQT_question construction. The extraction of word or letter is common
    word = re.findall("'(.*?)'", generated_template['sparql_wikidata'])[0]
    if generated_template['template_id'] == 1:
        # Word matching
        questions_generator.recreate_nnqt_question(generated_template, "Give me |property_0||entity_0| that contains the word {" + word + "} in their name", [0], [])
    else:
        # Starting letter matching
        questions_generator.recreate_nnqt_question(generated_template, "Give me |property_0||entity_0| that starts with {'" + word + "'}", [0], [])
    # Change NNQT_question format if property is not "instance of", otherwise there wouldn't be differences between the original and the modified questions
    query_property = questions_generator.get_elements_from_query(generated_template['sparql_wikidata'], [0], True)[0]
    if query_property == "P31":
        generated_template['NNQT_question'] = generated_template['NNQT_question'].replace("|property_0|", "")
    else:
        generated_template['NNQT_question'] = generated_template['NNQT_question'].replace("|property_0|", "{" + questions_generator.get_entity_name_from_wikidata_id(query_property) + "} ")

# Replace "mode of transport" with an entity that is not a person and which subclasses don't contain the term 'vehicle'
def string_matching_simple_contains_word_entity_example_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entity_id: str = "Q334166", old_entity: str = "mode of transport") -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    questions_generator.entity_example_generation_common_part(current_uid, generated_template, generated_questions, old_entity_id, old_entity, \
        wikidata_ids_extractor.get_wikidata_not_person_types())
    string_matching_simple_contains_word_nnqt_question_construction(generated_template)
    return generated_template

# Replace the known entity with a random candidate
def string_matching_simple_contains_word_entity_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    questions_generator.entity_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    string_matching_simple_contains_word_nnqt_question_construction(generated_template)
    return generated_template

# Replace "instance of" property with a random one. It should remain unchanged since it's a fixed property, but its substitution doesn't transform the question
# type in another one, and moreover also the question doesn't follow the template (because of the lack of the first triple)
def string_matching_simple_contains_word_relation_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [0], True)
    questions_generator.relation_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    string_matching_simple_contains_word_nnqt_question_construction(generated_template)
    return generated_template

# Replace the known entity with another entity of the same instance or subclass type, that makes the query unable to return results
def string_matching_simple_contains_word_entity_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    found, old_entities = questions_generator.entity_2_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    if not found:
        # There aren't valid candidates, so try with a random entity
        return string_matching_simple_contains_word_entity_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        string_matching_simple_contains_word_nnqt_question_construction(generated_template)
        return generated_template

# Replace "instance of" property with another property of the same instance or superproperty type, that makes the query unable to return results
def string_matching_simple_contains_word_relation_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [0], True)
    found, old_properties = questions_generator.relation_2_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    if not found:
        # There aren't valid candidates, so try with a random property
        return string_matching_simple_contains_word_relation_generation(current_uid, question_template, generated_questions, old_properties_ids, old_properties)
    else:
        string_matching_simple_contains_word_nnqt_question_construction(generated_template)
        return generated_template

# Replace the object entity with another entity of the same instance or subclass type, taken from the other properties of the subject entity, so the new
# query becomes unable to return results
def string_matching_simple_contains_word_entity_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    old_answer_id = next(iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values()))['value'].split("/")[-1]
    found, old_entities = questions_generator.entity_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, [
        "select ?ans ?ansLabel where {wd:" + old_answer_id + " ?rel ?ans . ?ans wdt:|rel_entity_type| wd:|entity_type| . ?ans rdfs:label ?ansLabel . " + \
            "FILTER (LANG(?ansLabel) = \"en\" && ?ans not in (wd:|old_entity_id|))} LIMIT 20"])
    if not found:
        # There aren't valid candidates, so try with a random entity of the same type or class
        return string_matching_simple_contains_word_entity_2_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        string_matching_simple_contains_word_nnqt_question_construction(generated_template)
        return generated_template

# Replace "instance of" property with another property that links the answer with another entity of the same type or class of the known entity
def string_matching_simple_contains_word_relation_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [0], True)
    print(sparql_query)
    old_answer_id = next(iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values()))['value'].split("/")[-1]
    found, old_properties = questions_generator.relation_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_properties_ids, \
        ["select ?ans where {wd:" + old_answer_id + " ?ans ?obj . ?obj wdt:|rel_entity_type| wd:|entity_type| . FILTER (?ans not in (wdt:|old_property_id|))} LIMIT 20"])
    if not found:
        # There aren't valid candidates, so try with "relation_2" function
        return string_matching_simple_contains_word_relation_2_generation(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    else:
        string_matching_simple_contains_word_nnqt_question_construction(generated_template)
        return generated_template
