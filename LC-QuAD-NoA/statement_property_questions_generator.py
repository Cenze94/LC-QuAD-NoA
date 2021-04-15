import questions_generator
import wikidata_ids_extractor

from typing import List, Dict, Any, Tuple
from copy import copy, deepcopy
import re

# Most of required data for NNQT_question construction is common among all operations of the same template
def statement_property_nnqt_question_construction(generated_template: Dict[str, Any]):
    sparql_query = generated_template['old_sparql_wikidata']
    if 'filter' in sparql_query.lower():
        # Find second entity value, substituting the answer variable with the corresponding variable
        answer_var_name = re.findall(r'SELECT (\?\w*) WHERE', sparql_query, re.IGNORECASE)[0]
        entity_var_name = re.findall(r' (\?\w*) filter', sparql_query, re.IGNORECASE)[0]
        sparql_query_entity = sparql_query.replace(answer_var_name, entity_var_name, 1)
        fixed_entity = next(iter(questions_generator.get_sparql_query_results(sparql_query_entity)['results']['bindings'][0].values()))['value'].split("/")[-1]
        # If entity is a date, keep only year-month-day part
        if questions_generator.get_element_type(fixed_entity) == questions_generator.ElementType.date:
            fixed_entity = fixed_entity.split("T")[0]
        fixed_entities = [fixed_entity]
        questions_generator.recreate_nnqt_question(generated_template, "What is |property_0| of |entity_0| that is |property_1| is |element_0| ?", [0], [0], False,
        fixed_entities = fixed_entities)
    else:
        questions_generator.recreate_nnqt_question(generated_template, "What is |property_0| of |entity_0| that is |property_1| is |entity_1| ?", [0, 1], [0], False)

# Replace "Somalia" with a city without 2009 population data
def statement_property_entity_example_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entity_id: str = "Q1045", old_entity: str = "Somalia") -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    questions_generator.entity_example_generation_common_part(current_uid, generated_template, generated_questions, old_entity_id, old_entity, \
        [wikidata_ids_extractor.DataType.cities])
    statement_property_nnqt_question_construction(generated_template)
    return generated_template

# Replace one of the known entities with a random candidate
def statement_property_entity_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        # Check if there is a filter: if not so the last element is an entity
        if 'filter' in sparql_query.lower():
            old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
        else:
            old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    questions_generator.entity_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    statement_property_nnqt_question_construction(generated_template)
    return generated_template

# There are two possible properties that can be substituted ({population}, which has to be replaced twice, and {point in time})
def statement_property_relation_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_specific_elements_from_query(sparql_query, [0], "p", "P")
        old_properties_ids += questions_generator.get_specific_elements_from_query(sparql_query, [0], "pq", "P")
    questions_generator.relation_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    statement_property_nnqt_question_construction(generated_template)
    return generated_template

# Replace the known entity with another entity of the same instance or subclass type, that makes the query unable to return results. If there isn't a filter
# then there is a second known entity
def statement_property_entity_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        # Check if there is a filter: if not so the last element is an entity
        if 'filter' in sparql_query.lower():
            old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
        else:
            old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    found, old_entities = questions_generator.entity_2_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    if not found:
        # There aren't valid candidates, so try with a random entity
        return statement_property_entity_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        statement_property_nnqt_question_construction(generated_template)
        return generated_template

# Replace one property with another property of the same instance or superproperty type, that makes the query unable to return results
def statement_property_relation_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_specific_elements_from_query(sparql_query, [0], "p", "P")
        old_properties_ids += questions_generator.get_specific_elements_from_query(sparql_query, [0], "pq", "P")
    found, old_properties = questions_generator.relation_2_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    if not found:
        # There aren't valid candidates, so try with a random property
        return statement_property_relation_generation(current_uid, question_template, generated_questions, old_properties_ids, old_properties)
    else:
        statement_property_nnqt_question_construction(generated_template)
        return generated_template

# Replace the subject entity with another entity of the same instance or subclass type, linked to the same original value through the same property. In this
# case there are two solutions, depending on the presence of "filter" in the original query
def statement_property_entity_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    # Check if there is a filter: if not so the last element is an entity
    query_contains_filter = 'filter' in sparql_query.lower()
    if query_contains_filter:
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    else:
        # If there isn't a filter then the last element is an entity
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    # In this case the procedure is different from normal: it takes the answer query, modifies it to obtain all entities of the same type or class of the
    # original answer that return results, and then uses this list to exclude these entities from possible candidates
    general_sparql_query = sparql_query.replace("wd:" + old_entities_ids[0], "?ans")
    # Get the substring between "{" and "}"
    general_sparql_query = re.findall("{(.+)}", general_sparql_query)[0]
    type_common_string = "?ans wdt:|rel_entity_type| wd:|entity_type|"
    """first_query = "SELECT distinct ?ans ?ansLabel WHERE {" + type_common_string + " . ?ans rdfs:label ?ansLabel . FILTER(LANG(?ansLabel) = \"en\" " + \
        "&& NOT EXISTS {" + type_common_string + " . " + general_sparql_query + "})} LIMIT 20" """
    # This version of the query is more complete and adapted for "entity_3" logic, but is also slow and unsafe, since sometimes raises a server error
    old_answer = next(iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values()))['value'].split("/")[-1]
    # Find answer filter and type
    old_answer_filter, _ = questions_generator.get_filter_from_element(old_answer, "obj", "s", False)
    first_query = "SELECT distinct ?ans ?ansLabel WHERE {" + type_common_string + " . ?ans ?rel ?obj . ?ans rdfs:label ?ansLabel . FILTER(LANG(?ansLabel) = \"en\" && " + \
        old_answer_filter + "NOT EXISTS {" + type_common_string + " . " + general_sparql_query + "})} LIMIT 5" 
    if query_contains_filter:
        found, old_entities = questions_generator.entity_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, [first_query])
    else:
        # In this case there is an additional normal query with two triples that link the first known entity to the second one
        found, old_entities = questions_generator.entity_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, [first_query,
            "select ?ans ?ansLabel where {wd:" + old_entities_ids[0] + " ?rel ?s . ?s ?rel2 ?ans . ?ans wdt:|rel_entity_type| wd:|entity_type| . ?ans rdfs:label ?ansLabel . " + \
                "FILTER (LANG(?ansLabel) = \"en\" && REGEX(STR(?s), \"Q(\\\\d+)-\") && ?ans not in (wd:|old_entity_id|))} LIMIT 20"])
    if not found:
        # There aren't valid candidates, so try with a random entity of the same type or class
        return statement_property_entity_2_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        statement_property_nnqt_question_construction(generated_template)
        return generated_template

# Replace the first or the last property with another property that links the answer with another entity of the same type or class of the known entity. The second
# property is ignored because is always equal to the first one, and so it's replaced when the first property is modified. Also in this case there are two solutions,
# depending on the presence of "filter" in the original query
def statement_property_relation_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    # Check if there is a filter: if not so the last element is an entity
    query_contains_filter = 'filter' in sparql_query.lower()
    if query_contains_filter:
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
        # There is only one entity used for both properties
        old_entities_ids.append(old_entities_ids[0])
    else:
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    old_properties_ids = [questions_generator.get_specific_elements_from_query(sparql_query, [0], "p", "P")[0]]
    old_properties_ids.append(questions_generator.get_specific_elements_from_query(sparql_query, [0], "pq", "P")[0])
    old_answer = next(iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values()))['value'].split("/")[-1]
    # Find answer filter and type
    old_answer_filter, element_type = questions_generator.get_filter_from_element(old_answer, "obj", "s")
    # The first query is the same for both cases
    if element_type == questions_generator.ElementType.entity:
        first_query = "select distinct ?ans where {?sbj ?ans ?s . ?s ?rel2 wd:" + old_answer + " . ?sbj wdt:|rel_entity_type| wd:|entity_type| . FILTER (" + \
            old_answer_filter + " && ?ans not in (p:|old_property_id|))} LIMIT 20"
    else:
        # The answer is not an entity
        first_query = "SELECT distinct ?ans WHERE {?sbj ?ans ?s . ?s ?rel2 ?obj . ?sbj wdt:|rel_entity_type| wd:|entity_type| . FILTER (" + old_answer_filter + \
            " && ?ans not in (p:|old_property_id|))} LIMIT 10"
    if query_contains_filter:
        # Find qualifier value, substituting the answer variable with the corresponding variable
        answer_var_name = re.findall(r'SELECT (\?\w*) WHERE', sparql_query, re.IGNORECASE)[0]
        qualifier_var_name = re.findall(old_properties_ids[1] + r' (\?\w*) filter', sparql_query, re.IGNORECASE)[0]
        sparql_query_qualifier = sparql_query.replace(answer_var_name, qualifier_var_name, 1)
        old_qualifier_value = next(iter(questions_generator.get_sparql_query_results(sparql_query_qualifier)['results']['bindings'][0].values()))['value'].split("/")[-1]
        # Find qualifier filter
        old_qualifier_filter, _ = questions_generator.get_filter_from_element(old_qualifier_value, "x", "s")
        # Since the answer is not an entity, in this case the queries results link the known entity to a value of the same type of the original one: the type is
        # defined through the corresponding filter. Besides the second query accepts only properties that are "qualifiers", so that are represented with the "pq" prefix
        found, old_properties = questions_generator.relation_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_properties_ids, \
            [first_query, "SELECT distinct ?ans WHERE { ?sbj ?rel ?s . ?s ?ans ?x . ?sbj wdt:|rel_entity_type| wd:|entity_type| . FILTER (" + old_qualifier_filter + \
                " && ?ans not in (pq:|old_property_id|)) } LIMIT 10"],
            ["", "pq"])
    else:
        # The first query is identical to the other case, the second instead becomes more normal
        found, old_properties = questions_generator.relation_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_properties_ids, \
            [first_query, "select distinct ?ans where {wd:" + old_entities_ids[0] + " ?rel ?s . ?s ?ans ?obj . ?obj wdt:|rel_entity_type| wd:|entity_type| . FILTER " + \
                "(REGEX(STR(?s), \"Q(\\\\d+)-\") && ?ans not in (pq:|old_property_id|))} LIMIT 20"],
            ["", "pq"])
    if not found:
        # There aren't valid candidates, so try with "relation_2" function
        return statement_property_relation_2_generation(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    else:
        statement_property_nnqt_question_construction(generated_template)
        return generated_template

# Most of required data for NNQT_question construction is common among all operations of the same template
def statement_property_2_nnqt_question_construction(generated_template: Dict[str, Any]):
    sparql_query = generated_template['old_sparql_wikidata']
    if 'filter' in sparql_query.lower():
        # Find second entity value, substituting the answer variable with the corresponding variable
        answer_var_name = re.findall(r'SELECT (\?\w*) WHERE', sparql_query, re.IGNORECASE)[0]
        entity_var_name = re.findall(r' (\?\w*) filter', sparql_query, re.IGNORECASE)[0]
        sparql_query_entity = sparql_query.replace(answer_var_name, entity_var_name, 1)
        fixed_entity = next(iter(questions_generator.get_sparql_query_results(sparql_query_entity)['results']['bindings'][0].values()))['value'].split("/")[-1]
        # If entity is a date, keep only year-month-day part
        if questions_generator.get_element_type(fixed_entity) == questions_generator.ElementType.date:
            fixed_entity = fixed_entity.split("T")[0]
        fixed_entities = [fixed_entity]
        questions_generator.recreate_nnqt_question(generated_template, "What is the |property_1| for |entity_0| has |property_0| as |element_0| ?", [0], [0], False,
        fixed_entities = fixed_entities)
    else:
        questions_generator.recreate_nnqt_question(generated_template, "What is the |property_1| for |entity_0| has |property_0| as |entity_1| ?", [0, 1], [0], False)

# Replace "The Shining" with a person that hasn't "1980" as "publication date"
def statement_property_2_entity_example_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entity_id: str = "Q186341", old_entity: str = "The Shining") -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    questions_generator.entity_example_generation_common_part(current_uid, generated_template, generated_questions, old_entity_id, old_entity, \
        wikidata_ids_extractor.get_wikidata_person_types())
    statement_property_2_nnqt_question_construction(generated_template)
    return generated_template

# Replace one of the known entities with a random candidate
def statement_property_2_entity_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        # Check if there is a filter: if not so the last element is an entity
        if 'filter' in sparql_query.lower():
            old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
        else:
            old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    questions_generator.entity_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    statement_property_2_nnqt_question_construction(generated_template)
    return generated_template

# There are two possible properties that can be substituted ({publication date}, which has to be replaced twice, and {place of publication})
def statement_property_2_relation_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_specific_elements_from_query(sparql_query, [0], "p", "P")
        old_properties_ids += questions_generator.get_specific_elements_from_query(sparql_query, [0], "pq", "P")
    questions_generator.relation_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    statement_property_2_nnqt_question_construction(generated_template)
    return generated_template

# Replace the known entity with another entity of the same instance or subclass type, that makes the query unable to return results. If there isn't a filter
# then there is a second known entity
def statement_property_2_entity_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        # Check if there is a filter: if not so the last element is an entity
        if 'filter' in sparql_query.lower():
            old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
        else:
            old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    found, old_entities = questions_generator.entity_2_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    if not found:
        # There aren't valid candidates, so try with a random entity
        return statement_property_2_entity_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        statement_property_2_nnqt_question_construction(generated_template)
        return generated_template

# Replace one property with another property of the same instance or superproperty type, that makes the query unable to return results
def statement_property_2_relation_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_specific_elements_from_query(sparql_query, [0], "p", "P")
        old_properties_ids += questions_generator.get_specific_elements_from_query(sparql_query, [0], "pq", "P")
    found, old_properties = questions_generator.relation_2_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    if not found:
        # There aren't valid candidates, so try with a random property
        return statement_property_2_relation_generation(current_uid, question_template, generated_questions, old_properties_ids, old_properties)
    else:
        statement_property_2_nnqt_question_construction(generated_template)
        return generated_template

# Replace the subject entity with another entity of the same instance or subclass type, linked to the same original value through the same property. In this
# case there are two solutions, depending on the presence of "filter" in the original query
def statement_property_2_entity_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    # Check if there is a filter: if not so the last element is an entity
    query_contains_filter = 'filter' in sparql_query.lower()
    if query_contains_filter:
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    else:
        # If there isn't a filter then the last element is an entity
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    # In this case the procedure is different from normal: it takes the answer query, modifies it to obtain all entities of the same type or class of the
    # original answer that return results, and then uses this list to exclude these entities from possible candidates
    general_sparql_query = sparql_query.replace("wd:" + old_entities_ids[0], "?ans")
    # Get the substring between "{" and "}"
    general_sparql_query = re.findall("{(.+)}", general_sparql_query)[0]
    type_common_string = "?ans wdt:|rel_entity_type| wd:|entity_type|"
    """first_query = "SELECT distinct ?ans ?ansLabel WHERE {" + type_common_string + " . ?ans rdfs:label ?ansLabel . FILTER(LANG(?ansLabel) = \"en\" " + \
        "&& NOT EXISTS {" + type_common_string + " . " + general_sparql_query + "})} LIMIT 20" """
    # This version of the query is more complete and adapted for "entity_3" logic, but is also slow and unsafe, since sometimes raises a server error
    old_answer = next(iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values()))['value'].split("/")[-1]
    # Find answer filter and type
    old_answer_filter, _ = questions_generator.get_filter_from_element(old_answer, "obj", "s", False)
    first_query = "SELECT distinct ?ans ?ansLabel WHERE {" + type_common_string + " . ?ans ?rel ?obj . ?ans rdfs:label ?ansLabel . FILTER(LANG(?ansLabel) = \"en\" && " + \
        old_answer_filter + "NOT EXISTS {" + type_common_string + " . " + general_sparql_query + "})} LIMIT 5" 
    if query_contains_filter:
        found, old_entities = questions_generator.entity_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, [first_query])
    else:
        # In this case there is an additional normal query with two triples that link the first known entity to the second one
        found, old_entities = questions_generator.entity_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, [first_query,
            "select ?ans ?ansLabel where {wd:" + old_entities_ids[0] + " ?rel ?s . ?s ?rel2 ?ans . ?ans wdt:|rel_entity_type| wd:|entity_type| . ?ans rdfs:label ?ansLabel . " + \
                "FILTER (LANG(?ansLabel) = \"en\" && REGEX(STR(?s), \"Q(\\\\d+)-\") && ?ans not in (wd:|old_entity_id|))} LIMIT 20"])
    if not found:
        # There aren't valid candidates, so try with a random entity of the same type or class
        return statement_property_2_entity_2_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        statement_property_2_nnqt_question_construction(generated_template)
        return generated_template

# Replace the first or the last property with another property that links the answer with another entity of the same type or class of the known entity. The second
# property is ignored because is always equal to the first one, and so it's replaced when the first property is modified. Also in this case there are two solutions,
# depending on the presence of "filter" in the original query
def statement_property_2_relation_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    # Check if there is a filter: if not so the last element is an entity
    query_contains_filter = 'filter' in sparql_query.lower()
    if query_contains_filter:
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
        # There is only one entity used for both properties
        old_entities_ids.append(old_entities_ids[0])
    else:
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    old_properties_ids = [questions_generator.get_specific_elements_from_query(sparql_query, [0], "p", "P")[0]]
    # The order is inverted because in the queries of this case the first entity is linked to the second property and the second entity is linked to the first property
    old_properties_ids.insert(0, questions_generator.get_specific_elements_from_query(sparql_query, [0], "pq", "P")[0])
    old_answer = next(iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values()))['value'].split("/")[-1]
    # Find answer filter and type
    old_answer_filter, element_type = questions_generator.get_filter_from_element(old_answer, "obj", "s")
    # The first query is the same for both cases
    if element_type == questions_generator.ElementType.entity:
        first_query = "select distinct ?ans where {?sbj ?rel ?s . ?s ?ans wd:" + old_answer + " . ?sbj wdt:|rel_entity_type| wd:|entity_type| . FILTER (" + \
            old_answer_filter + " && ?ans not in (pq:|old_property_id|))} LIMIT 20"
    else:
        # The answer is not an entity
        first_query = "SELECT distinct ?ans WHERE {?sbj ?rel ?s . ?s ?ans ?obj . ?sbj wdt:|rel_entity_type| wd:|entity_type| . FILTER (" + old_answer_filter + \
            " && ?ans not in (pq:|old_property_id|))} LIMIT 10"
    if query_contains_filter:
        # Find qualifier value, substituting the answer variable with the corresponding variable
        answer_var_name = re.findall(r'SELECT (\?\w*) WHERE', sparql_query, re.IGNORECASE)[0]
        qualifier_var_name = re.findall(old_properties_ids[1] + r' (\?\w*) filter', sparql_query, re.IGNORECASE)[0]
        sparql_query_qualifier = sparql_query.replace(answer_var_name, qualifier_var_name, 1)
        old_qualifier_value = next(iter(questions_generator.get_sparql_query_results(sparql_query_qualifier)['results']['bindings'][0].values()))['value'].split("/")[-1]
        # Find qualifier filter
        old_qualifier_filter, _ = questions_generator.get_filter_from_element(old_qualifier_value, "x", "s")
        # Since the answer is not an entity, in this case the queries results link the known entity to a value of the same type of the original one: the type is
        # defined through the corresponding filter. Besides the second query accepts only properties that are "qualifiers", so that are represented with the "p" prefix
        found, old_properties = questions_generator.relation_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_properties_ids, \
            [first_query, "SELECT distinct ?ans WHERE { ?sbj ?ans ?s . ?s ?rel2 ?x . ?sbj wdt:|rel_entity_type| wd:|entity_type| . FILTER (" + old_qualifier_filter + \
                " && ?ans not in (p:|old_property_id|)) } LIMIT 10"],
            ["pq", ""])
    else:
        # The first query is identical to the other case, the second instead becomes more normal
        found, old_properties = questions_generator.relation_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_properties_ids, \
            [first_query, "select distinct ?ans where {wd:" + old_entities_ids[0] + " ?ans ?s . ?s ?rel2 ?obj . ?obj wdt:|rel_entity_type| wd:|entity_type| . FILTER " + \
                "(REGEX(STR(?s), \"Q(\\\\d+)-\") && ?ans not in (p:|old_property_id|))} LIMIT 20"],
            ["pq", ""])
    if not found:
        # There aren't valid candidates, so try with "relation_2" function
        return statement_property_2_relation_2_generation(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    else:
        statement_property_2_nnqt_question_construction(generated_template)
        return generated_template