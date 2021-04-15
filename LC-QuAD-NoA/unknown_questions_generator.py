import questions_generator
import wikidata_ids_extractor

from typing import List, Dict, Any, Tuple
from copy import copy, deepcopy
import re

# Most of required data for NNQT_question construction is common among all operations of the same template
def unknown_nnqt_question_construction(generated_template: Dict[str, Any]):
    questions_generator.recreate_nnqt_question(generated_template, "What is |property_0| of |entity_0| and |property_1|", [0], [0], False)

# Replace "Seattle" with a city without head of government
def unknown_entity_example_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entity_id: str = "Q5083", old_entity: str = "Seattle") -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    questions_generator.entity_example_generation_common_part(current_uid, generated_template, generated_questions, old_entity_id, old_entity, \
        [wikidata_ids_extractor.DataType.cities])
    unknown_nnqt_question_construction(generated_template)
    return generated_template

# Replace the known entity with a random candidate
def unknown_entity_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    questions_generator.entity_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    unknown_nnqt_question_construction(generated_template)
    return generated_template

# There are two possible properties that can be substituted ({head of government}, which has to be replaced twice, and {work period (end)})
def unknown_relation_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_specific_elements_from_query(sparql_query, [0], "p", "P")
        old_properties_ids += questions_generator.get_specific_elements_from_query(sparql_query, [0], "pq", "P")
    questions_generator.relation_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    unknown_nnqt_question_construction(generated_template)
    return generated_template

# Replace the known entity with another entity of the same instance or subclass type, that makes the query unable to return results
def unknown_entity_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    found, old_entities = questions_generator.entity_2_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    if not found:
        # There aren't valid candidates, so try with a random entity
        return unknown_entity_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        unknown_nnqt_question_construction(generated_template)
        return generated_template

# Replace one property with another property of the same instance or superproperty type, that makes the query unable to return results
def unknown_relation_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_specific_elements_from_query(sparql_query, [0], "p", "P")
        old_properties_ids += questions_generator.get_specific_elements_from_query(sparql_query, [0], "pq", "P")
    found, old_properties = questions_generator.relation_2_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    if not found:
        # There aren't valid candidates, so try with a random property
        return unknown_relation_generation(current_uid, question_template, generated_questions, old_properties_ids, old_properties)
    else:
        unknown_nnqt_question_construction(generated_template)
        return generated_template

# Replace the subject entity with another entity of the same instance or subclass type, linked to the same original value through the same property
def unknown_entity_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    # In this case the procedure is different from normal: it takes the answer query, modifies it to obtain all entities of the same type or class of the
    # original answer that return results, and then uses this list to exclude these entities from possible candidates
    general_sparql_query = sparql_query.replace("wd:" + old_entities_ids[0], "?ans")
    # Get the substring between "{" and "}"
    general_sparql_query = re.findall("{(.+)}", general_sparql_query)[0]
    type_common_string = "?ans wdt:|rel_entity_type| wd:|entity_type|"
    """query = "SELECT distinct ?ans ?ansLabel WHERE {" + type_common_string + " . ?ans rdfs:label ?ansLabel . FILTER(LANG(?ansLabel) = \"en\" " + \
        "&& NOT EXISTS {" + type_common_string + " . " + general_sparql_query + "})} LIMIT 20" """
    # This version of the query is more complete and adapted for "entity_3" logic, but is also slow and unsafe, since sometimes raises a server error
    old_answer = next(iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values()))['value'].split("/")[-1]
    # Find answer filter and type
    old_answer_filter, _ = questions_generator.get_filter_from_element(old_answer, "obj", "s", False)
    query = "SELECT distinct ?ans ?ansLabel WHERE {" + type_common_string + " . ?ans ?rel ?obj . ?ans rdfs:label ?ansLabel . FILTER(LANG(?ansLabel) = \"en\" && " + \
        old_answer_filter + "NOT EXISTS {" + type_common_string + " . " + general_sparql_query + "})} LIMIT 5" 
    found, old_entities = questions_generator.entity_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, [query])
    if not found:
        # There aren't valid candidates, so try with a random entity of the same type or class
        return unknown_entity_2_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        unknown_nnqt_question_construction(generated_template)
        return generated_template

# Replace the first or the last property with another property that links the answer with another entity of the same type or class of the known entity. The second
# property is ignored because is always equal to the first one, and so it's replaced when the first property is modified
def unknown_relation_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    # There is only one entity used for both properties
    old_entities_ids.append(old_entities_ids[0])
    old_properties_ids = questions_generator.get_specific_elements_from_query(sparql_query, [0], "p", "P")
    old_properties_ids += questions_generator.get_specific_elements_from_query(sparql_query, [0], "pq", "P")
    # Answers are inverted
    old_answers = iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values())
    old_answer_2 = next(old_answers)['value'].split("/")[-1]
    old_answer_1 = next(old_answers)['value'].split("/")[-1]
    # Find answers filter and type
    old_answer_1_filter, element_type_1 = questions_generator.get_filter_from_element(old_answer_1, "obj", "s")
    old_answer_2_filter, element_type_2 = questions_generator.get_filter_from_element(old_answer_2, "obj", "s")
    if element_type_1 == questions_generator.ElementType.entity:
        first_query = "select distinct ?ans where {?sbj ?ans ?s . ?s ?rel2 wd:" + old_answer_1 + " . ?sbj wdt:|rel_entity_type| wd:|entity_type| . FILTER (" + \
            old_answer_1_filter + " && ?ans not in (p:|old_property_id|))} LIMIT 20"
    else:
        # If the answer is not an entity, in this case the queries results link the known entity to a value of the same type of the original one: the type is
        # defined through the corresponding filter
        first_query = "SELECT distinct ?ans WHERE {?sbj ?ans ?s . ?s ?rel2 ?obj . ?sbj wdt:|rel_entity_type| wd:|entity_type| . FILTER (" + old_answer_1_filter + \
            " && ?ans not in (p:|old_property_id|))} LIMIT 10"
    if element_type_2 == questions_generator.ElementType.entity:
        second_query = "select distinct ?ans where {?sbj ?rel ?s . ?s ?ans wd:" + old_answer_2 + " . ?sbj wdt:|rel_entity_type| wd:|entity_type| . FILTER (" + \
            old_answer_2_filter + " && ?ans not in (pq:|old_property_id|))} LIMIT 20"
    else:
        # If the answer is not an entity, in this case the queries results link the known entity to a value of the same type of the original one: the type is
        # defined through the corresponding filter
        second_query = "SELECT distinct ?ans WHERE {?sbj ?rel ?s . ?s ?ans ?obj . ?sbj wdt:|rel_entity_type| wd:|entity_type| . FILTER (" + old_answer_2_filter + \
            " && ?ans not in (pq:|old_property_id|))} LIMIT 10"
    # The second query accepts only properties that are "qualifiers", so that are represented with the "pq" prefix
    found, old_properties = questions_generator.relation_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_properties_ids, \
        [first_query, second_query], ["", "pq"])
    if not found:
        # There aren't valid candidates, so try with "relation_2" function
        return unknown_relation_2_generation(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    else:
        unknown_nnqt_question_construction(generated_template)
        return generated_template

# Most of required data for NNQT_question construction is common among all operations of the same template
def unknown_2_nnqt_question_construction(generated_template: Dict[str, Any]):
    questions_generator.recreate_nnqt_question(generated_template, "What is |property_1| and |property_2| of |entity_0| has |property_0| as |entity_1|", \
        [0, 1], [0, 1], False)

# Replace "Lothair I" with a city without head of government
def unknown_2_entity_example_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entity_id: str = "Q150735", old_entity: str = "Lothair I") -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    questions_generator.entity_example_generation_common_part(current_uid, generated_template, generated_questions, old_entity_id, old_entity, \
        [wikidata_ids_extractor.DataType.politicians_and_leaders])
    unknown_2_nnqt_question_construction(generated_template)
    return generated_template

# Replace one of the known entities with a random candidate
def unknown_2_entity_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    questions_generator.entity_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    unknown_2_nnqt_question_construction(generated_template)
    return generated_template

# There are three possible properties that can be substituted ({noble title}, which has to be replaced twice, {follows} and {followed by})
def unknown_2_relation_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_specific_elements_from_query(sparql_query, [0], "p", "P")
        old_properties_ids += questions_generator.get_specific_elements_from_query(sparql_query, [0, 1], "pq", "P")
    questions_generator.relation_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    unknown_2_nnqt_question_construction(generated_template)
    return generated_template

# Replace one of the known entities with another entity of the same instance or subclass type, that makes the query unable to return results
def unknown_2_entity_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    found, old_entities = questions_generator.entity_2_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    if not found:
        return unkown_2_entity_generation(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    else:
        unknown_2_nnqt_question_construction(generated_template)
        return generated_template

# Replace one property with another property of the same instance or superproperty type, that makes the query unable to return results
def unknown_2_relation_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_specific_elements_from_query(sparql_query, [0], "p", "P")
        old_properties_ids += questions_generator.get_specific_elements_from_query(sparql_query, [0, 1], "pq", "P")
    found, old_properties = questions_generator.relation_2_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    if not found:
        # There aren't valid candidates, so try with a random property
        return unknown_2_relation_generation(current_uid, question_template, generated_questions, old_properties_ids, old_properties)
    else:
        unknown_2_nnqt_question_construction(generated_template)
        return generated_template

# Replace the first subject entity with another entity of the same instance or subclass type, linked to a value of the same type of the original one through
# the same property. The chosen value is the first by convention. The same is applied to the second object, linked to the first subject
def unknown_2_entity_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    # In this case the procedure is different from normal: it takes the answer query, modifies it to obtain all entities of the same type or class of the
    # original answer that return results, and then uses this list to exclude these entities from possible candidates
    general_sparql_query = sparql_query.replace("wd:" + old_entities_ids[0], "?ans")
    # Get the substring between "{" and "}"
    general_sparql_query = re.findall("{(.+)}", general_sparql_query)[0]
    type_common_string = "?ans wdt:|rel_entity_type| wd:|entity_type|"
    """query = "SELECT distinct ?ans ?ansLabel WHERE {" + type_common_string + " . ?ans rdfs:label ?ansLabel . FILTER(LANG(?ansLabel) = \"en\" " + \
        "&& NOT EXISTS {" + type_common_string + " . " + general_sparql_query + "})} LIMIT 20" """
    # This version of the query is more complete and adapted for "entity_3" logic, but is also slow and unsafe, since sometimes raises a server error
    old_answer = next(iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values()))['value'].split("/")[-1]
    # Find answer filter and type
    old_answer_filter, _ = questions_generator.get_filter_from_element(old_answer, "obj", "s", False)
    query = "SELECT distinct ?ans ?ansLabel WHERE {" + type_common_string + " . ?ans ?rel ?obj . ?ans rdfs:label ?ansLabel . FILTER(LANG(?ansLabel) = \"en\" && " + \
        old_answer_filter + "NOT EXISTS {" + type_common_string + " . " + general_sparql_query + "})} LIMIT 5"
    found, old_entities = questions_generator.entity_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, [
        query, "select ?ans ?ansLabel where {wd:" + old_entities_ids[0] + " ?rel ?s . ?s ?rel2 ?ans . ?ans wdt:|rel_entity_type| wd:|entity_type| . " + \
            "?ans rdfs:label ?ansLabel . FILTER (LANG(?ansLabel) = \"en\" && REGEX(STR(?s), \"Q(\\\\d+)-\") && ?ans not in (wd:|old_entity_id|))} LIMIT 20"
    ])
    if not found:
        # There aren't valid candidates, so try with a random entity of the same type or class
        return unknown_2_entity_2_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        unknown_2_nnqt_question_construction(generated_template)
        return generated_template

# Replace the first or the last property with another property that links the answer with another entity of the same type or class of the known entity. The second
# property is ignored because is always equal to the first one, and so it's replaced when the first property is modified
def unknown_2_relation_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    # The first entity is used for the last two properties
    old_entities_ids.reverse()
    old_entities_ids.append(old_entities_ids[0])
    old_properties_ids = questions_generator.get_specific_elements_from_query(sparql_query, [0], "p", "P")
    old_properties_ids += questions_generator.get_specific_elements_from_query(sparql_query, [0, 1], "pq", "P")
    # Answers are inverted
    old_answers = iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values())
    old_answer_2 = next(old_answers)['value'].split("/")[-1]
    old_answer_1 = next(old_answers)['value'].split("/")[-1]
    # Find answers filter and type
    old_answer_1_filter, element_type_1 = questions_generator.get_filter_from_element(old_answer_1, "obj", "s")
    old_answer_2_filter, element_type_2 = questions_generator.get_filter_from_element(old_answer_2, "obj", "s")
    # These two queries are identical except for the associated property and the answer filter
    if element_type_1 == questions_generator.ElementType.entity:
        second_query = "select distinct ?ans where {?sbj ?rel ?s . ?s ?ans wd:" + old_answer_1 + " . ?sbj wdt:|rel_entity_type| wd:|entity_type| . FILTER (" + \
            old_answer_1_filter + " && ?ans not in (pq:|old_property_id|))} LIMIT 20"
    else:
        # If the answer is not an entity, in this case the queries results link the known entity to a value of the same type of the original one: the type is
        # defined through the corresponding filter
        second_query = "SELECT distinct ?ans WHERE {?sbj ?rel ?s . ?s ?ans ?obj . ?sbj wdt:|rel_entity_type| wd:|entity_type| . FILTER (" + old_answer_1_filter + \
            " && ?ans not in (pq:|old_property_id|))} LIMIT 10"
    if element_type_2 == questions_generator.ElementType.entity:
        third_query = "select distinct ?ans where {?sbj ?rel ?s . ?s ?ans wd:" + old_answer_2 + " . ?sbj wdt:|rel_entity_type| wd:|entity_type| . FILTER (" + \
            old_answer_2_filter + " && ?ans not in (pq:|old_property_id|))} LIMIT 20"
    else:
        # If the answer is not an entity, in this case the queries results link the known entity to a value of the same type of the original one: the type is
        # defined through the corresponding filter
        third_query = "SELECT distinct ?ans WHERE {?sbj ?rel ?s . ?s ?ans ?obj . ?sbj wdt:|rel_entity_type| wd:|entity_type| . FILTER (" + old_answer_2_filter + \
            " && ?ans not in (pq:|old_property_id|))} LIMIT 10"
    # The second query accepts only properties that are "qualifiers", so that are represented with the "pq" prefix
    found, old_properties = questions_generator.relation_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_properties_ids, \
        ["select distinct ?ans where {wd:" + old_entities_ids[1] + " ?ans ?s . ?s ?rel2 ?obj . ?obj wdt:|rel_entity_type| wd:|entity_type| . FILTER " + \
            "(REGEX(STR(?s), \"Q(\\\\d+)-\") && ?ans not in (p:|old_property_id|))} LIMIT 20", second_query, third_query], ["", "pq", "pq"])
    if not found:
        # There aren't valid candidates, so try with "relation_2" function
        return unknown_2_relation_2_generation(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    else:
        unknown_2_nnqt_question_construction(generated_template)
        return generated_template