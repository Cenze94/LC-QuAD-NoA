import questions_generator
import wikidata_ids_extractor

from typing import List, Dict, Any, Tuple
from copy import copy, deepcopy
import random
import re

# Most of required data for NNQT_question construction is common among all operations of the same template
def right_subgraph_nnqt_question_construction(generated_template: Dict[str, Any]):
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
        elif 'Point(' in fixed_entity:
            # This is a coordinate, extract the value considered in the old NNQT_question
            fixed_entity = re.findall("'(.*?)'", sparql_query)[0]
        fixed_entities = [fixed_entity]
        questions_generator.recreate_nnqt_question(generated_template, "What is |property_0| of |entity_0|, that has |property_1| is |element_0| ?", [0], [0, 1],
        fixed_entities = fixed_entities)
    else:
        questions_generator.recreate_nnqt_question(generated_template, "What is |property_0| of |entity_0|, that has |property_1| is |entity_1| ?", [0, 1], [0, 1])

# Replace "Albert I, Prince of Monaco" with a politician or leader entity that isn't linked to any "place of burial" of "Cathedral of Our Lady Immaculate"
def right_subgraph_entity_example_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entity_id: str = "Q159646", old_entity: str = "Albert I, Prince of Monaco") -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    questions_generator.entity_example_generation_common_part(current_uid, generated_template, generated_questions, old_entity_id, old_entity, \
        [wikidata_ids_extractor.DataType.politicians_and_leaders])
    right_subgraph_nnqt_question_construction(generated_template)
    return generated_template

# Replace one of the known entities with a random candidate
def right_subgraph_entity_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    if old_entities_ids is None:
        if 'filter' in sparql_query.lower():
            old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
        else:
            old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    questions_generator.entity_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    right_subgraph_nnqt_question_construction(generated_template)
    return generated_template

# There are two possible properties that can be substituted ({father} and {place of burial}), the first is present in every version of the question,
# while the second is present only in "NNQT_question"
def right_subgraph_relation_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    if old_properties_ids is None:
        old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1], True)
    questions_generator.relation_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    right_subgraph_nnqt_question_construction(generated_template)
    return generated_template

# Create a new valid right subgraph question
def right_subgraph_template_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], \
    str, str, str, str, str]:
    generated_template = deepcopy(question_template)
    # There could be properties without an English label, in that case repeat the whole operation; this should be a very rare case
    while True:
        # A generated entity possibly couldn't have properties, although it should happen rarely: in that case a new random entity is obtained and the entire operation is repeated
        while True:
            # First entity is the subject; obj2 will still be extracted, because it could not have an English label
            first_entity_id, first_entity = wikidata_ids_extractor.get_random_wikidata_entity_from_all()
            possible_questions_data = questions_generator.get_sparql_query_results("select distinct ?rel ?rel2 ?obj ?objLabel ?obj2 ?obj2Label where {wd:" + \
                first_entity_id + " ?rel ?obj . ?obj ?rel2 ?obj2 . ?obj rdfs:label ?objLabel . ?obj2 rdfs:label ?obj2Label . FILTER (LANG(?objLabel) = \"en\" &&" + \
                    " LANG(?obj2Label) = \"en\")} LIMIT 10")
            possible_questions_number = len(possible_questions_data['results']['bindings'])
            if possible_questions_number > 0:
                break
        possible_questions_random_index = random.choice(range(possible_questions_number))
        question_data = possible_questions_data['results']['bindings'][possible_questions_random_index]
        obj_entity_id = question_data['obj']['value'].split('/')[-1]
        rel_property_id = question_data['rel']['value'].split('/')[-1]
        rel2_property_id = question_data['rel2']['value'].split('/')[-1]
        obj2_entity_id = question_data['obj2']['value'].split('/')[-1]
        try:
            # Get entities and relation English names
            obj_entity = question_data['objLabel']['value'].replace("_", " ")
            rel_property = questions_generator.get_entity_name_from_wikidata_id(rel_property_id)
            rel2_property = questions_generator.get_entity_name_from_wikidata_id(rel2_property_id)
            obj2_entity = question_data['obj2Label']['value'].replace("_", " ")
            # Verify if the first answer got with the query is the subject found during question generation. If the query has a lot of results is probable that the first answer will be
            # in a position different from the first
            generated_template['sparql_wikidata'] = "select distinct ?obj where { wd:" + first_entity_id + " wdt:" + rel_property_id + " ?obj . ?obj " + rel2_property_id + \
                " wd:" + obj2_entity_id + " . ?obj rdfs:label ?objLabel . FILTER (LANG(?objLabel) = \"en\") } LIMIT 5"
            sparql_result = next(iter(questions_generator.get_sparql_query_results(generated_template['sparql_wikidata'])['results']['bindings'][0].values()))['value'].split("/")[-1]
            if sparql_result == sbj_entity_id:
                break
        except:
            continue
    nnqt_question = "What is the {" + rel_property + "} for {" + first_entity + "}, that has {" + rel2_property + "} is {" + obj2_entity + "} ?"
    generated_template['NNQT_question'] = nnqt_question
    generated_template['question'] = nnqt_question
    generated_template['paraphrased_question'] = nnqt_question
    return generated_template, first_entity_id, first_entity, rel_property_id, rel_property

# Replace one of the known entities with another entity of the same instance or subclass type, that makes the query unable to return results
def right_subgraph_entity_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    if old_entities_ids is None:
        if 'filter' in sparql_query.lower():
            old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
        else:
            old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    found, old_entities = questions_generator.entity_2_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    if not found:
        # There aren't valid candidates, so try with a random entity
        return right_subgraph_entity_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        right_subgraph_nnqt_question_construction(generated_template)
        return generated_template

# Replace one property with another property of the same instance or superproperty type, that makes the query unable to return results
def right_subgraph_relation_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    if old_properties_ids is None:
        old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1], True)
    found, old_properties = questions_generator.relation_2_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids)
    if not found:
        # There aren't valid candidates, so try with a random property
        return right_subgraph_relation_generation(current_uid, question_template, generated_questions, old_properties_ids, old_properties)
    else:
        right_subgraph_nnqt_question_construction(generated_template)
        return generated_template

# Replace the subject entity with another entity of the same instance or subclass type, taken from the other properties of the object entity, so the new
# query becomes unable to return results. The same can be done taking the second object entity and the properties of the second subject. The choice between
# the two options, when both are feasible, is random
def right_subgraph_entity_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    if 'filter' in sparql_query.lower():
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    else:
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    old_answer_id = next(iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values()))['value'].split("/")[-1]
    found, old_entities = questions_generator.entity_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, [
        "select ?ans ?ansLabel where {?ans ?rel wd:" + old_answer_id + " . ?ans wdt:|rel_entity_type| wd:|entity_type| . ?ans rdfs:label ?ansLabel . " + \
            "FILTER (LANG(?ansLabel) = \"en\" && ?ans not in (wd:|old_entity_id|))} LIMIT 20",
        "select ?ans ?ansLabel where {wd:" + old_answer_id + " ?rel ?ans . ?ans wdt:|rel_entity_type| wd:|entity_type| . ?ans rdfs:label ?ansLabel . " + \
            "FILTER (LANG(?ansLabel) = \"en\" && ?ans not in (wd:|old_entity_id|))} LIMIT 20"])
    if not found:
        # There aren't valid candidates, so try with a random entity of the same type or class
        return right_subgraph_entity_2_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        right_subgraph_nnqt_question_construction(generated_template)
        return generated_template

# Replace one of the properties with another property that links the answer with another entity of the same type or class of the known entity
def right_subgraph_relation_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    query_contains_filter = 'filter' in sparql_query.lower()
    if query_contains_filter:
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
        # Find second entity value, substituting the answer variable with the corresponding variable
        answer_var_name = re.findall(r'SELECT (\?\w*) WHERE', sparql_query, re.IGNORECASE)[0]
        entity_var_name = re.findall(r' (\?\w*) filter', sparql_query, re.IGNORECASE)[0]
        sparql_query_entity = sparql_query.replace(answer_var_name, entity_var_name, 1)
        old_entities_ids.append(next(iter(questions_generator.get_sparql_query_results(sparql_query_entity)['results']['bindings'][0].values()))['value'].split("/")[-1])
    else:
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1], True)
    old_answer_id = next(iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values()))['value'].split("/")[-1]
    if query_contains_filter:
        second_query = "select ?ans where {wd:" + old_answer_id + " ?ans ?obj . FILTER (|filter|?obj| && ?ans not in (wdt:|old_property_id|))} LIMIT 20"
    else:
        second_query = "select ?ans where {wd:" + old_answer_id + " ?ans ?obj . ?obj wdt:|rel_entity_type| wd:|entity_type| . FILTER (?ans not in (wdt:|old_property_id|))} LIMIT 20"
    found, old_properties = questions_generator.relation_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_properties_ids, \
        ["select ?ans where {?sbj ?ans wd:" + old_answer_id + " . ?sbj wdt:|rel_entity_type| wd:|entity_type| . FILTER (?ans not in (wdt:|old_property_id|))} LIMIT 20",
        second_query])
    if not found:
        # There aren't valid candidates, so try with "relation_2" function
        return right_subgraph_relation_2_generation(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    else:
        right_subgraph_nnqt_question_construction(generated_template)
        return generated_template

# Most of required data for NNQT_question construction is common among all operations of the same template
def right_subgraph_2_nnqt_question_construction(generated_template: Dict[str, Any]):
    questions_generator.recreate_nnqt_question(generated_template, "What is |property_1| of |property_0| of |entity_0| ?", [0], [0, 1])

# Replace "Ebola hemorrhagic fever" with a politician or leader entity that isn't linked to any "location of discovery" which is "member of" any entity
def right_subgraph_2_entity_example_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entity_id: str = "Q51993", old_entity: str = "Ebola hemorrhagic fever") -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    questions_generator.entity_example_generation_common_part(current_uid, generated_template, generated_questions, old_entity_id, old_entity, \
        [wikidata_ids_extractor.DataType.politicians_and_leaders])
    right_subgraph_2_nnqt_question_construction(generated_template)
    return generated_template

# Replace the known entity with a random candidate
def right_subgraph_2_entity_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    questions_generator.entity_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    right_subgraph_2_nnqt_question_construction(generated_template)
    return generated_template

# There are two possible properties that can be substituted ({location or discovery} and {member of}), replace one of them with a random property
def right_subgraph_2_relation_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1], True)
    questions_generator.relation_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    right_subgraph_2_nnqt_question_construction(generated_template)
    return generated_template

# Replace one of the known entities with another entity of the same instance or subclass type, that makes the query unable to return results
def right_subgraph_2_entity_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    found, old_entities = questions_generator.entity_2_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    if not found:
        # There aren't valid candidates, so try with a random entity
        return right_subgraph_2_entity_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        right_subgraph_2_nnqt_question_construction(generated_template)
        return generated_template

# Replace one property with another property of the same instance or superproperty type, that makes the query unable to return results
def right_subgraph_2_relation_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1], True)
    found, old_properties = questions_generator.relation_2_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    if not found:
        # There aren't valid candidates, so try with a random property
        return right_subgraph_2_relation_generation(current_uid, question_template, generated_questions, old_properties_ids, old_properties)
    else:
        right_subgraph_2_nnqt_question_construction(generated_template)
        return generated_template

# Replace the subject entity with another entity of the same instance or subclass type, taken from the other properties of the object entity, so the new
# query becomes unable to return results
def right_subgraph_2_entity_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    answer_var_name = re.findall(r'SELECT (\?\w*) WHERE', sparql_query, re.IGNORECASE)[0]
    entity_var_name = re.findall(r'. (\?\w*) wdt:', sparql_query)[0]
    modified_sparql_query = sparql_query.replace(answer_var_name, entity_var_name, 1)
    modified_answer_entity = questions_generator.get_sparql_query_results(modified_sparql_query)['results']['bindings'][0][entity_var_name[1:]]['value'].split("/")[-1]
    modified_answer_filter, _ = questions_generator.get_filter_from_element(modified_answer_entity, "obj", "", False)
    if modified_answer_filter:
        query = "select ?ans ?ansLabel where {?ans ?rel ?obj . ?ans wdt:|rel_entity_type| wd:|entity_type| . ?ans rdfs:label ?ansLabel . " + \
            "FILTER (" + modified_answer_filter + "LANG(?ansLabel) = \"en\" && ?ans not in (wd:|old_entity_id|))} LIMIT 20"
    else:
        query = "select ?ans ?ansLabel where {?ans ?rel wd:" + modified_answer_entity + " . ?ans wdt:|rel_entity_type| wd:|entity_type| . ?ans rdfs:label ?ansLabel . " + \
            "FILTER (LANG(?ansLabel) = \"en\" && ?ans not in (wd:|old_entity_id|))} LIMIT 20"
    found, old_entities = questions_generator.entity_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, [query])
    if not found:
        # There aren't valid candidates, so try with a random entity of the same type or class
        return right_subgraph_2_entity_2_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        right_subgraph_2_nnqt_question_construction(generated_template)
        return generated_template

# Replace one of the properties with another property that links the answer with another entity of the same type or class of the known entity. In the
# second triple there isn't a known entity, so the query is modified and executed to find a suitable entity
def right_subgraph_2_relation_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    answer_var_name = re.findall(r'SELECT (\?\w*) WHERE', sparql_query, re.IGNORECASE)[0]
    entity_var_name = re.findall(r'. (\?\w*) wdt:', sparql_query)[0]
    modified_sparql_query = sparql_query.replace(answer_var_name, entity_var_name, 1)
    modified_answer_entity = questions_generator.get_sparql_query_results(modified_sparql_query)['results']['bindings'][0][entity_var_name[1:]]['value'].split("/")[-1]
    old_entities_ids.append(modified_answer_entity)
    old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1], True)
    old_answer_id = next(iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values()))['value'].split("/")[-1]
    modified_answer_filter, _ = questions_generator.get_filter_from_element(modified_answer_entity, "obj", "", False)
    old_answer_filter, _ = questions_generator.get_filter_from_element(old_answer_id, "sbj", "", False)
    if modified_answer_filter:
        first_query = "select ?ans where {?sbj ?ans ?obj . ?sbj wdt:|rel_entity_type| wd:|entity_type| . FILTER (" + modified_answer_filter + "?ans not in (wdt:|old_property_id|))} LIMIT 20"
    else:
        first_query = "select ?ans where {?sbj ?ans wd:" + modified_answer_entity + " . ?sbj wdt:|rel_entity_type| wd:|entity_type| . FILTER (?ans not in (wdt:|old_property_id|))} LIMIT 20"
    if old_answer_filter:
        second_query = "select ?ans where {?sbj ?ans ?obj . ?obj wdt:|rel_entity_type| wd:|entity_type| . FILTER (" + old_answer_filter + "?ans not in (wdt:|old_property_id|))} LIMIT 20"
    else:
        second_query = "select ?ans where {wd:" + old_answer_id + " ?ans ?obj . ?obj wdt:|rel_entity_type| wd:|entity_type| . FILTER (?ans not in (wdt:|old_property_id|))} LIMIT 20"
    found, old_properties = questions_generator.relation_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_properties_ids, \
        [first_query, second_query])
    if not found:
        # There aren't valid candidates, so try with "relation_2" function
        return right_subgraph_2_relation_2_generation(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    else:
        right_subgraph_2_nnqt_question_construction(generated_template)
        return generated_template