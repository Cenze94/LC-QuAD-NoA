import questions_generator
import wikidata_ids_extractor

from typing import List, Dict, Any, Tuple
from copy import copy, deepcopy

# Most of required data for NNQT_question construction is common among all operations of the same template
def two_intentions_right_subgraph_nnqt_question_construction(generated_template: Dict[str, Any]):
    questions_generator.recreate_nnqt_question(generated_template, "What is the |property_0| and the |property_1| of |entity_0| ?", [0], [0, 1])

# Replace "John Denver" with a person entity that hasn't a "cause of death" or a "place of death"
def two_intentions_right_subgraph_entity_example_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entity_id: str = "Q105460", old_entity: str = "John Denver") -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    questions_generator.entity_example_generation_common_part(current_uid, generated_template, generated_questions, old_entity_id, old_entity, \
        wikidata_ids_extractor.get_wikidata_person_types())
    two_intentions_right_subgraph_nnqt_question_construction(generated_template)
    return generated_template

# Replace the known entity with a random candidate
def two_intentions_right_subgraph_entity_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    questions_generator.entity_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    two_intentions_right_subgraph_nnqt_question_construction(generated_template)
    return generated_template

# There are two possible properties that can be substituted ({cause of death} and {place of death})
def two_intentions_right_subgraph_relation_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1], True)
    questions_generator.relation_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    two_intentions_right_subgraph_nnqt_question_construction(generated_template)
    return generated_template

# Replace one of the known entities with another entity of the same instance or subclass type, that makes the query unable to return results. Since the subject of the two
# triples is the same, only the first entity is considered
def two_intentions_right_subgraph_entity_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    found, old_entities = questions_generator.entity_2_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    if not found:
        # There aren't valid candidates, so try with a random entity
        return two_intentions_right_subgraph_entity_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        two_intentions_right_subgraph_nnqt_question_construction(generated_template)
        return generated_template

# Replace one property with another property of the same instance or superproperty type, that makes the query unable to return results
def two_intentions_right_subgraph_relation_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1], True)
    found, old_properties = questions_generator.relation_2_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    if not found:
        # There aren't valid candidates, so try with a random property
        return two_intentions_right_subgraph_relation_generation(current_uid, question_template, generated_questions, old_properties_ids, old_properties)
    else:
        two_intentions_right_subgraph_nnqt_question_construction(generated_template)
        return generated_template

# Replace the subject entity with another entity of the same instance or subclass type, taken from the other properties of the object entity, so the new
# query becomes unable to return results. Since the subject of the two triples is the same, only the first entity is considered
def two_intentions_right_subgraph_entity_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    answers_iter = iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values())
    old_answer_id_1 = next(answers_iter)['value'].split("/")[-1]
    old_answer_id_2 = next(answers_iter)['value'].split("/")[-1]
    found, old_entities = questions_generator.entity_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, [
        "select distinct ?ans ?ansLabel where {?ans ?rel wd:" + old_answer_id_1 + " . ?ans wdt:|rel_entity_type| wd:|entity_type| . ?ans rdfs:label ?ansLabel . " + \
            "FILTER (LANG(?ansLabel) = \"en\" && ?ans not in (wd:|old_entity_id|))} LIMIT 20", "select distinct ?ans ?ansLabel where {?ans ?rel wd:" + old_answer_id_2 + \
                " . ?ans wdt:|rel_entity_type| wd:|entity_type| . ?ans rdfs:label ?ansLabel . FILTER (LANG(?ansLabel) = \"en\" && ?ans not in (wd:|old_entity_id|))} LIMIT 20"])
    if not found:
        # There aren't valid candidates, so try with a random entity of the same type or class
        return two_intentions_right_subgraph_entity_2_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        two_intentions_right_subgraph_nnqt_question_construction(generated_template)
        return generated_template

# Replace one of the properties with another property that links the answer with another entity of the same type or class of the known entity. Since there are two different
# answers, the two queries differs for the answer id added to them
def two_intentions_right_subgraph_relation_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1], True)
    answers_iter = iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values())
    old_answer_id_1 = next(answers_iter)['value'].split("/")[-1]
    old_answer_id_2 = next(answers_iter)['value'].split("/")[-1]
    found, old_properties = questions_generator.relation_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_properties_ids, \
        ["select ?ans where {?sbj ?ans wd:" + old_answer_id_1 + " . ?sbj wdt:|rel_entity_type| wd:|entity_type| . FILTER (?ans not in (wdt:|old_property_id|))} LIMIT 20",
        "select ?ans where {?sbj ?ans wd:" + old_answer_id_2 + " . ?sbj wdt:|rel_entity_type| wd:|entity_type| . FILTER (?ans not in (wdt:|old_property_id|))} LIMIT 20"])
    if not found:
        # There aren't valid candidates, so try with "relation_2" function
        return two_intentions_right_subgraph_relation_2_generation(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    else:
        two_intentions_right_subgraph_nnqt_question_construction(generated_template)
        return generated_template