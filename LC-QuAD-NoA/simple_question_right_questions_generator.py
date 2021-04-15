import questions_generator
import wikidata_ids_extractor

from typing import List, Dict, Any, Tuple, Union
from copy import copy, deepcopy

# Most of required data for NNQT_question construction is common among all operations of the same template
def simple_question_right_nnqt_question_construction(generated_template: Dict[str, Any]):
    questions_generator.recreate_nnqt_question(generated_template, "What is the |entity_1| for |property_0| of |entity_0|", [0, 1], [0])

# Substitute "John Jay" with a politician or leader entity that hasn't a "position held" linked to any "public office"
def simple_question_right_entity_example_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entity_id: str = "Q310847", old_entity: str = "John Jay") -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    questions_generator.entity_example_generation_common_part(current_uid, generated_template, generated_questions, old_entity_id, old_entity, \
        [wikidata_ids_extractor.DataType.politicians_and_leaders])
    simple_question_right_nnqt_question_construction(generated_template)
    return generated_template

# Replace one of the known entities with a random candidate
def simple_question_right_entity_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    questions_generator.entity_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    simple_question_right_nnqt_question_construction(generated_template)
    return generated_template

# Substitute "position held" with another relation that doesn't link "John Jay" to any "public office"
def simple_question_right_relation_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [0], True)
    questions_generator.relation_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    simple_question_right_nnqt_question_construction(generated_template)
    return generated_template

# Replace one of the known entities with another entity of the same instance or subclass type, that makes the query unable to return results
def simple_question_right_entity_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    found, old_entities = questions_generator.entity_2_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    if not found:
        # There aren't valid candidates, so try with a random entity
        return simple_question_right_entity_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        simple_question_right_nnqt_question_construction(generated_template)
        return generated_template


# Replace the first property with another property of the same instance or superproperty type, that makes the query unable to return results.
# The "instance of" property at the end is ignored, since otherwise the template would change
def simple_question_right_relation_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [0], True)
    found, old_properties = questions_generator.relation_2_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    if not found:
        # There aren't valid candidates, so try with a random property
        return simple_question_right_relation_generation(current_uid, question_template, generated_questions, old_properties_ids, old_properties)
    else:
        simple_question_right_nnqt_question_construction(generated_template)
        return generated_template

# Replace one of the known entities with another entity of the same instance or subclass type, taken from the other properties of the unknown entity, so the new
# query becomes unable to return results
def simple_question_right_entity_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    old_answer_id = next(iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values()))['value'].split("/")[-1]
    found, old_entities = questions_generator.entity_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, [
        "select ?ans ?ansLabel where {?ans ?rel wd:" + old_answer_id + " . ?ans wdt:|rel_entity_type| wd:|entity_type| . ?ans rdfs:label ?ansLabel . FILTER " + \
            "(LANG(?ansLabel) = \"en\" && ?ans not in (wd:|old_entity_id|))} LIMIT 20",
            "select ?ans ?ansLabel where {wd:" + old_answer_id + " ?rel ?ans . ?ans wdt:|rel_entity_type| wd:|entity_type| . ?ans rdfs:label ?ansLabel . " + \
                "FILTER (LANG(?ansLabel) = \"en\" && ?ans not in (wd:|old_entity_id|))} LIMIT 20"])
    if not found:
        # There aren't valid candidates, so try with a random entity of the same type or class
        return simple_question_right_entity_2_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        simple_question_right_nnqt_question_construction(generated_template)
        return generated_template

# Replace the first property with another property that links the answer with another entity of the same type or class of the object entity.
# The "instance of" property at the end is ignored, since otherwise the template would change
def simple_question_right_relation_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [0], True)
    old_answer_id = next(iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values()))['value'].split("/")[-1]
    found, old_properties = questions_generator.relation_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_properties_ids, \
        ["select ?ans where {?sbj ?ans wd:" + old_answer_id + " . ?sbj wdt:|rel_entity_type| wd:|entity_type| . FILTER (?ans not in (wdt:|old_property_id|))} LIMIT 20"])
    if not found:
        # There aren't valid candidates, so try with "relation_2" function
        return simple_question_right_relation_2_generation(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    else:
        simple_question_right_nnqt_question_construction(generated_template)
        return generated_template