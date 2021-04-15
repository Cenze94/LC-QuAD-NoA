import questions_generator
import wikidata_ids_extractor

from typing import List, Dict, Any, Tuple, Union
from copy import copy, deepcopy
import random

# Most of required data for NNQT_question construction is common among all operations of the same template
def simple_question_left_nnqt_question_construction(generated_template: Dict[str, Any]):
    questions_generator.recreate_nnqt_question(generated_template, "What is the |entity_1| for |property_0| of |entity_0|", [0, 1], [0])

# Substitute "Mahmoud Abbas" with a politician or leader entity that isn't a "head of state" of any "country"
def simple_question_left_entity_example_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entity_id: str = "Q127998", old_entity: str = "Mahmoud Abbas") -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    questions_generator.entity_example_generation_common_part(current_uid, generated_template, generated_questions, old_entity_id, old_entity, \
        [wikidata_ids_extractor.DataType.politicians_and_leaders])
    simple_question_left_nnqt_question_construction(generated_template)
    return generated_template

# Replace one of the known entities with a random candidate
def simple_question_left_entity_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    questions_generator.entity_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    simple_question_left_nnqt_question_construction(generated_template)
    return generated_template

# Substitute "head of state" with another relation that doesn't link any entity of type "country" to "Mahmoud Abbas"
def simple_question_left_relation_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [0], True)
    questions_generator.relation_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    simple_question_left_nnqt_question_construction(generated_template)
    return generated_template

# Create a new valid simple question left
def simple_question_left_template_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], \
    str, str, str, str, str]:
    generated_template = deepcopy(question_template)
    # There could be properties without an English label, in that case repeat the whole operation; this should be a very rare case
    while True:
        # A generated entity possibly couldn't have properties, although it should happen rarely: in that case a new random entity is obtained and the entire operation is repeated
        while True:
            # First entity is the object, I made this decision because I saw that doing so is easier to find a valid solution; sbj will still be extracted, because it could
            # not have an English label
            first_entity_id, first_entity = wikidata_ids_extractor.get_random_wikidata_entity_from_all()
            possible_questions_data = questions_generator.get_sparql_query_results("select distinct ?rel ?sbj ?sbjLabel ?objio ?objioLabel where {" +
                "?sbj ?rel wd:" + first_entity_id + " . ?sbj wdt:P31 ?objio . ?sbj rdfs:label ?sbjLabel . ?objio rdfs:label ?objioLabel ." + \
                    " FILTER (LANG(?sbjLabel) = \"en\" && LANG(?objioLabel) = \"en\")} LIMIT 10")
            possible_questions_number = len(possible_questions_data['results']['bindings'])
            if possible_questions_number > 0:
                break
        possible_questions_random_index = random.choice(range(possible_questions_number))
        question_data = possible_questions_data['results']['bindings'][possible_questions_random_index]
        sbj_entity_id = question_data['sbj']['value'].split('/')[-1]
        rel_property_id = question_data['rel']['value'].split('/')[-1]
        objio_entity_id = question_data['objio']['value'].split('/')[-1]
        try:
            # Get entities and relation English names
            sbj_entity = question_data['sbjLabel']['value'].replace("_", " ")
            rel_property = questions_generator.get_entity_name_from_wikidata_id(rel_property_id)
            objio_entity = question_data['objioLabel']['value'].replace("_", " ")
            # Verify if the first answer got with the query is the subject found during question generation. If the query has a lot of results is probable that the first answer will be
            # in a position different from the first
            generated_template['sparql_wikidata'] = "select distinct ?sbj where { ?sbj wdt:" + rel_property_id + " wd:" + first_entity_id + " . ?sbj wdt:P31 wd:" + objio_entity_id + \
                " . ?sbj rdfs:label ?sbjLabel . FILTER (LANG(?sbjLabel) = \"en\") } LIMIT 5"
            sparql_result = next(iter(questions_generator.get_sparql_query_results(generated_template['sparql_wikidata'])['results']['bindings'][0].values()))['value'].split("/")[-1]
            if sparql_result == sbj_entity_id:
                break
        except:
            continue
    nnqt_question = "What is the {" + objio_entity + "} for {" + rel_property + "} of {" + first_entity + "}"
    generated_template['NNQT_question'] = nnqt_question
    generated_template['question'] = nnqt_question
    generated_template['paraphrased_question'] = nnqt_question
    return generated_template, first_entity_id, first_entity, rel_property_id, rel_property

# Create a new valid simple question left, then substitute one entity or the property to make it invalid
def simple_question_left_generic_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template, first_entity_id, first_entity, rel_property_id, rel_property = simple_question_left_template_generation(current_uid, question_template, generated_questions)
    # True question generated, now it has to be invalidated for the purpose of this updated dataset. Since both entity and relation invalidation are accepted,
    # choose randomly one of them
    operation_type_boolean = random.randint(0, 1)
    if operation_type_boolean == 0:
        updated_template = simple_question_left_entity_generation(current_uid, generated_template, generated_questions, first_entity_id, first_entity)
        return updated_template
    else:
        updated_template = simple_question_left_relation_generation(current_uid, generated_template, generated_questions, rel_property_id, rel_property)
        return updated_template

# Replace one of the known entities with another entity of the same instance or subclass type, that makes the query unable to return results
def simple_question_left_entity_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_entities_ids: List[str] = None, old_entities: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_entities_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    found, old_entities = questions_generator.entity_2_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_entities)
    if not found:
        # There aren't valid candidates, so try with a random entity
        return simple_question_left_entity_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        simple_question_left_nnqt_question_construction(generated_template)
        return generated_template

# Replace the first property with another property of the same instance or superproperty type, that makes the query unable to return results.
# The "instance of" property at the end is ignored, since otherwise the template would change
def simple_question_left_relation_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]],
old_properties_ids: List[str] = None, old_properties: List[str] = None) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    if old_properties_ids is None:
        sparql_query = generated_template['sparql_wikidata']
        old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [0], True)
    found, old_properties = questions_generator.relation_2_generation_common_part(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    if not found:
        # There aren't valid candidates, so try with a random property
        return simple_question_left_relation_generation(current_uid, question_template, generated_questions, old_properties_ids, old_properties)
    else:
        simple_question_left_nnqt_question_construction(generated_template)
        return generated_template

# Replace one of the known entities with another entity of the same instance or subclass type, taken from the other properties of the unknown entity, so the new
# query becomes unable to return results
def simple_question_left_entity_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0, 1])
    old_answer_id = next(iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values()))['value'].split("/")[-1]
    found, old_entities = questions_generator.entity_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, [
        "select ?ans ?ansLabel where {wd:" + old_answer_id + " ?rel ?ans . ?ans wdt:|rel_entity_type| wd:|entity_type| . ?ans rdfs:label ?ansLabel . FILTER " + \
            "(LANG(?ansLabel) = \"en\" && ?ans not in (wd:|old_entity_id|))} LIMIT 20",
            "select ?ans ?ansLabel where {wd:" + old_answer_id + " ?rel ?ans . ?ans wdt:|rel_entity_type| wd:|entity_type| . ?ans rdfs:label ?ansLabel . " + \
                "FILTER (LANG(?ansLabel) = \"en\" && ?ans not in (wd:|old_entity_id|))} LIMIT 20"])
    if not found:
        # There aren't valid candidates, so try with a random entity of the same type or class
        return simple_question_left_entity_2_generation(current_uid, question_template, generated_questions, old_entities_ids, old_entities)
    else:
        simple_question_left_nnqt_question_construction(generated_template)
        return generated_template

# Replace the first property with another property that links the answer with another entity of the same type or class of the object entity.
# The "instance of" property at the end is ignored, since otherwise the template would change
def simple_question_left_relation_3_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated_template = deepcopy(question_template)
    sparql_query = generated_template['sparql_wikidata']
    old_entities_ids = questions_generator.get_elements_from_query(sparql_query, [0])
    old_properties_ids = questions_generator.get_elements_from_query(sparql_query, [0], True)
    old_answer_id = next(iter(questions_generator.get_sparql_query_results(sparql_query)['results']['bindings'][0].values()))['value'].split("/")[-1]
    found, old_properties = questions_generator.relation_3_generation_common_part(current_uid, generated_template, generated_questions, old_entities_ids, old_properties_ids, \
        ["select ?ans where {wd:" + old_answer_id + " ?ans ?obj . ?obj wdt:|rel_entity_type| wd:|entity_type| . FILTER (?ans not in (wdt:|old_property_id|))} LIMIT 20"])
    if not found:
        # There aren't valid candidates, so try with "relation_2" function
        return simple_question_left_relation_2_generation(current_uid, generated_template, generated_questions, old_properties_ids, old_properties)
    else:
        simple_question_left_nnqt_question_construction(generated_template)
        return generated_template

# Create a new valid simple question left, then substitute one entity or the property to make it invalid. This time the methods used for invalidation are "entity_3" and "relation_3",
# moreover an additional "both" option is added, which allows the generation of 2 questions from the same generated template, one with a modified entity and one with a different property
def simple_question_left_generic_2_generation(current_uid: int, question_template: Dict[str, Any], generated_questions: List[Dict[str, Any]], both = False) -> Union[Dict[str, Any],
List[Dict[str, Any]]]:
    # True question generated, now it has to be invalidated for the purpose of this updated dataset
    generated_template, first_entity_id, first_entity, rel_property_id, rel_property = simple_question_left_template_generation(current_uid, question_template, generated_questions)
    if both:
        # Get both entity and relation invalidation versions
        final_questions = []
        final_questions.append(simple_question_left_entity_3_generation(current_uid, generated_template, generated_questions))
        final_questions.append(simple_question_left_relation_3_generation(current_uid, generated_template, generated_questions))
        return final_questions
    else:
        # Since both entity and relation invalidation are accepted, choose randomly one of them
        operation_type_boolean = random.randint(0, 1)
        if operation_type_boolean == 0:
            updated_template = simple_question_left_entity_3_generation(current_uid, generated_template, generated_questions)
            return updated_template
        else:
            updated_template = simple_question_left_relation_3_generation(current_uid, generated_template, generated_questions)
            return updated_template