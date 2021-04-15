from enum import Enum
from typing import List, Tuple
import json
import random

class DataType(Enum):
    religious_figures = "Religious_figures.json"
    politicians_and_leaders = "Politicians_and_leaders.json"
    countries = "Countries_filtered.json"
    cities = "Cities_filtered.json"
    professions = "Professions.json"
    sports = "Sports.json"

# Extract IDs from a specific Wikipedia Vital Articles Category JSON file. These files have been obtained
# with PetScan (https://petscan.wmflabs.org/)
def wikidata_ids_extractor(id_type: DataType) -> Tuple[List[str], List[str]]:
    with open("entities_and_properties/" + id_type.value, "r") as read_file:
        json_file = json.load(read_file)
    ids_original_list = json_file.get("*")[0].get("a").get("*")
    ids_list = []
    titles_list = []
    for id_data in ids_original_list:
        ids_list.append(id_data.get("q"))
        titles_list.append(id_data.get("title").replace("_", " "))
    return ids_list, titles_list

# Get a random entity ID and label
def get_random_wikidata_entity(id_type: DataType) -> Tuple[str, str]:
    ids_list, titles_list = wikidata_ids_extractor(id_type)
    random_element_index = random.choice(range(len(ids_list)))
    return ids_list[random_element_index], titles_list[random_element_index]

# Return a random property ID and label
def get_random_wikidata_property() -> Tuple[str, str]:
    with open("entities_and_properties/Properties.json", "r") as read_file:
        json_file = json.load(read_file)
    random_element = random.choice(json_file)
    return random_element['id'], random_element['label']

# Return a random ID and label from a list of JSON files, so merging all ids and titles into two unique lists
def get_random_wikidata_entity_from_list(json_list: List[DataType]) -> Tuple[str, str]:
    all_ids_list = []
    all_titles_list = []
    for element in json_list:
        ids_list, titles_list = wikidata_ids_extractor(element)
        all_ids_list.extend(ids_list)
        all_titles_list.extend(titles_list)
    random_element_index = random.choice(range(len(all_ids_list)))
    return all_ids_list[random_element_index], all_titles_list[random_element_index]

# Get a random entity ID and label from any category
def get_random_wikidata_entity_from_all() -> Tuple[str, str]:
    typesList = [e for e in DataType]
    return get_random_wikidata_entity_from_list(typesList)

# Get a random person entity ID and label
def get_random_wikidata_person_entity() -> Tuple[str, str]:
    typesList = [DataType.religious_figures, DataType.politicians_and_leaders]
    return get_random_wikidata_entity_from_list(typesList)

# Return person types
def get_wikidata_person_types() -> List[DataType]:
    return [DataType.religious_figures, DataType.politicians_and_leaders]

# Get a random entity ID and label that is not a person
def get_random_wikidata_not_person_entity() -> Tuple[str, str]:
    typesList = [DataType.professions, DataType.sports, DataType.cities, DataType.countries]
    return get_random_wikidata_entity_from_list(typesList)

# Return types not referring to people
def get_wikidata_not_person_types() -> List[DataType]:
    return [DataType.professions, DataType.sports, DataType.cities, DataType.countries]
