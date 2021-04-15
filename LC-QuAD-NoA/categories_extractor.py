import json
import questions_generator

# Extract categories from downloaded JSON files, like countries from "Countries.json"

def load_countries_list():
    with open("entities_and_properties/countries_list.txt", "r") as countries_list_file:
        lines = [line.rstrip('\n') for line in countries_list_file]
        return lines

def extract_countries_from_json():
    countries_list = load_countries_list()
    with open("entities_and_properties/Countries.json", "r") as countries_json_file:
        json_file = json.load(countries_json_file)
    ids_original_list = json_file.get("*")[0].get("a").get("*")
    for id_data in ids_original_list[:]:
        id_name = id_data.get("title")
        id_name = id_name.replace("_", " ")
        if not id_name in countries_list:
            ids_original_list.remove(id_data)
    with open("entities_and_properties/Countries_filtered.json", "w") as filtered_countries_json_file:
        json.dump(json_file, filtered_countries_json_file)

def extract_cities_from_json():
    with open("entities_and_properties/not_cities_list.txt", "r") as not_cities_list_file:
        not_cities_list = [line.rstrip('\n') for line in not_cities_list_file]
    with open("entities_and_properties/Cities.json", "r") as cities_json_file:
        json_file = json.load(cities_json_file)
    ids_original_list = json_file.get("*")[0].get("a").get("*")
    for id_data in ids_original_list[:]:
        id_name = id_data.get("title")
        id_name = id_name.replace("_", " ")
        if id_name in not_cities_list:
            ids_original_list.remove(id_data)
    with open("entities_and_properties/Cities_filtered.json", "w") as filtered_cities_json_file:
        json.dump(json_file, filtered_cities_json_file)

def filter_properties_from_json():
    with open("entities_and_properties/Properties.json", "r") as json_file:
        properties_list = json.load(json_file)
        for element in properties_list[:]:
            property_name = element['label'].lower()
            property_type = element['datatype'].lower()
            if ' id' in property_name or '_id' in property_name or '-id' in property_name or ' code' in property_name or \
            'index' in property_name or ' model' in property_name or '-id' in property_type or 'url' in property_type or \
            'rating' in property_name or 'code ' in property_name or ' url' in property_name:
                properties_list.remove(element)
    with open("entities_and_properties/Properties.json", "w") as filtered_json_file:
        json.dump(properties_list, filtered_json_file)

# Get professions IDs, get the respective labels and build the JSON file
def build_professions_json():
    with open("entities_and_properties/professions.txt", "r") as prof_file:
        prof_ids_list = [line.rstrip('\n') for line in prof_file]
    prof_list = []
    for prof in prof_ids_list:
        prof_title = questions_generator.get_entity_name_from_wikidata_id(prof)
        if prof_title != prof_title.upper():
            prof_title = prof_title.lower()
        print(prof_title)
        prof_element = {"title": prof_title, "q": prof}
        prof_list.append(prof_element)
    prof_dict = {"*": [{"a": {"*": prof_list}}]}
    with open("entities_and_properties/Professions.json", "w") as prof_json_file:
        json.dump(prof_dict, prof_json_file)

# Get the sports list and write JSON file with them
def get_sports_json():
    sports_query_results = questions_generator.get_sparql_query_results("select ?sbj ?sbjLabel {{ select ?sbj ?sbjLabel where {?sbj wdt:P31 wd:Q31629 . " +
        "SERVICE wikibase:label { bd:serviceParam wikibase:language \"en\". }}} FILTER (LANG(?sbjLabel) = \"en\") }")
    sports_list = []
    for element in sports_query_results['results']['bindings']:
        sport_id = element['sbj']['value'].split('/')[-1]
        sport_title = element['sbjLabel']['value'].lower()
        sport_element = {"title": sport_title, "q": sport_id}
        sports_list.append(sport_element)
    sport_dict = {"*": [{"a": {"*": sports_list}}]}
    with open("entities_and_properties/Sports.json", "w") as sport_json_file:
        json.dump(sport_dict, sport_json_file)

#extract_countries_from_json()
#extract_cities_from_json()
#filter_properties_from_json()
#build_professions_json()
get_sports_json()