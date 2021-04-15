import questions_generator
import wikidata_ids_extractor

from typing import List, Dict, Any, Tuple
from copy import copy, deepcopy
import random
import re

# Most of required data for NNQT_question construction is common among all operations of the same template
def rank_nnqt_question_construction(generated_template: Dict[str, Any]):
    if 'desc' in generated_template['sparql_wikidata'].lower():
        order_string = "MAX"
    else:
        order_string = "MIN"
    questions_generator.recreate_nnqt_question(generated_template, "What is the |entity_0| with the |property_0| ?", [0], [1])
    property_name = re.findall("{(.*?)}", generated_template['NNQT_question'])[1]
    generated_template['NNQT_question'] = generated_template['NNQT_question'].replace(property_name, order_string + "(" + property_name + ")")

# Replace "open cluster" with a person entity, which instances cannot have any "radius" property
def rank_entity_example_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entity_id: str = "Q11387", old_entity: str = "open cluster") -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    questions_generator.entity_example_generation_common_part(current_uid, generated_template, generated_questions, old_entity_id, old_entity, \
        wikidata_ids_extractor.get_wikidata_person_types())
    rank_nnqt_question_construction(generated_template)
    return generated_template

# Replace the known entity with a random candidate
def rank_entity_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    questions_generator.entity_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    rank_nnqt_question_construction(generated_template)
    return generated_template

# Replace "radius" with another relation that doesn't link the answer entity with any entity (considering that ordering is alphanumeric)
def rank_relation_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [1], True)
    questions_generator.relation_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    rank_nnqt_question_construction(generated_template)
    return generated_template

# Replace one of the known entities with another entity of the same instance or subclass type, that makes the query unable to return results
def rank_entity_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    found, old_entities = questions_generator.entity_2_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    if not found:
        # There aren't valid candidates, so try with a random entity
        return rank_entity_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        rank_nnqt_question_construction(generated_template)
        return generated_template

# Replace one property with another property of the same instance or superproperty type, that makes the query unable to return results
def rank_relation_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [1], True)
    found, old_properties = questions_generator.relation_2_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    if not found:
        # There aren't valid candidates, so try with a random property
        return rank_relation_generation(current_uid, question_template, generated_questions, old_properties_ids, old_properties)
    else:
        rank_nnqt_question_construction(generated_template)
        return generated_template

# Replace the object entity with another entity of the same instance or subclass type, taken from the other properties of the subject entity, so the new
# query becomes unable to return results
def rank_entity_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    old_answer_id = next(iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values()))['value'].split("/")[-1]
    found, old_entities = questions_generator.entity_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, [
        "select ?ans ?ansLabel where {wd:" + old_answer_id + " ?rel ?ans . ?ans wdt:|rel_entity_type| wd:|entity_type| . ?ans rdfs:label ?ansLabel . " + \
            "FILTER (LANG(?ansLabel) = \"en\" && ?ans not in (wd:|old_entity_id|))} LIMIT 20"])
    if not found:
        # There aren't valid candidates, so try with a random entity of the same type or class
        return rank_entity_2_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        rank_nnqt_question_construction(generated_template)
        return generated_template

# The first property is always "instance of", and the second is linked to an unknown element, so replace the first since modifying it wouldn't change the template type
def rank_relation_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [0], True)
    old_answer_id = next(iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values()))['value'].split("/")[-1]
    found, old_properties = questions_generator.relation_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_properties_ids, \
        ["select ?ans where {wd:" + old_answer_id + " ?ans ?obj . ?obj wdt:|rel_entity_type| wd:|entity_type| . FILTER (?ans not in (wdt:|old_property_id|))} LIMIT 20"])
    if not found:
        # There aren't valid candidates, so try with "relation_2" function. Don't pass "old_properties_ids" and "old_properties" as input, since the involved properties are different
        return rank_relation_2_generation(current_uid, generated_template, generated_questions)
    else:
        rank_nnqt_question_construction(generated_template)
        return generated_template

# Most of required data for NNQT_question construction is common among all operations of the same template
def rank_2_nnqt_question_construction(generated_template: Dict[str, Any]):
    if 'desc' in generated_template['sparql_wikidata'].lower():
        order_string = "MAX"
    else:
        order_string = "MIN"
    questions_generator.recreate_nnqt_question(generated_template, "What is the |entity_0| with the |property_0| whose |property_1| is |entity_1| ?", [0, 1], [1, 2])
    property_name = re.findall("{(.*?)}", generated_template['NNQT_question'])[1]
    generated_template['NNQT_question'] = generated_template['NNQT_question'].replace(property_name, order_string + "(" + property_name + ")")

# Replace "association football" with an other sport entity, without a specific "number of clubs" specified for any "region"
def rank_2_entity_example_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entity_id: str = "Q2736", old_entity: str = "association football") -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    questions_generator.entity_example_generation_common_part(current_uid, generated_template, generated_questions, old_entity_id, old_entity, \
        [wikidata_ids_extractor.DataType.sports])
    return generated_template

# Replace the known entity with a random candidate
def rank_2_entity_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    questions_generator.entity_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    rank_2_nnqt_question_construction(generated_template)
    return generated_template

# Replace a random known property with another relation that doesn't link any "sport in a geographical region" of type "association football" to other entities
def rank_2_relation_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [1, 2], True)
    questions_generator.relation_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    rank_2_nnqt_question_construction(generated_template)
    return generated_template

# Replace one of the known entities with another entity of the same instance or subclass type, that makes the query unable to return results
def rank_2_entity_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    found, old_entities = questions_generator.entity_2_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    if not found:
        # There aren't valid candidates, so try with a random entity
        return rank_2_entity_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        rank_2_nnqt_question_construction(generated_template)
        return generated_template

# Replace one property with another property of the same instance or superproperty type, that makes the query unable to return results
def rank_2_relation_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [1, 2], True)
    found, old_properties = questions_generator.relation_2_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    if not found:
        # There aren't valid candidates, so try with a random property
        return rank_2_relation_generation(current_uid, question_template, generated_questions, old_properties_ids, old_properties)
    else:
        rank_2_nnqt_question_construction(generated_template)
        return generated_template

# Replace the subject entity with another entity of the same instance or subclass type, taken from the other properties of the object entity, so the new
# query becomes unable to return results. The same can be done taking the second object entity and the properties of the second subject. The choice between
# the two options, when both are feasible, is random
def rank_2_entity_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    old_answer_id = next(iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values()))['value'].split("/")[-1]
    found, old_entities = questions_generator.entity_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, [
        "select ?ans ?ansLabel where {wd:" + old_answer_id + " ?rel ?ans . ?ans wdt:|rel_entity_type| wd:|entity_type| . ?ans rdfs:label ?ansLabel . " + \
            "FILTER (LANG(?ansLabel) = \"en\" && ?ans not in (wd:|old_entity_id|))} LIMIT 20",
        "select ?ans ?ansLabel where {wd:" + old_answer_id + " ?rel ?ans . ?ans wdt:|rel_entity_type| wd:|entity_type| . ?ans rdfs:label ?ansLabel . " + \
            "FILTER (LANG(?ansLabel) = \"en\" && ?ans not in (wd:|old_entity_id|))} LIMIT 20"])
    if not found:
        # There aren't valid candidates, so try with a random entity of the same type or class
        return rank_2_entity_2_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        rank_2_nnqt_question_construction(generated_template)
        return generated_template

# Replace the last property with another property that links the answer with another entity of the same type or class of the known entity. The other two properties are
# not available because the first must be "instance of" and the second is linked to a numeric value, and so not to an entity; besides this value is used only for ranking,
# so changing the property wouldn't make the question unanswerable
def rank_2_relation_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [1])
    old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [2], True)
    old_answer_id = next(iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values()))['value'].split("/")[-1]
    found, old_properties = questions_generator.relation_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_properties_ids, \
        ["select ?ans where {wd:" + old_answer_id + " ?ans ?obj . ?obj wdt:|rel_entity_type| wd:|entity_type| . FILTER (?ans not in (wdt:|old_property_id|))} LIMIT 20"])
    if not found:
        # There aren't valid candidates, so try with "relation_2" function
        return rank_2_relation_2_generation(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    else:
        rank_2_nnqt_question_construction(generated_template)
        return generated_template