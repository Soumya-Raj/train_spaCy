import configparser

config = configparser.ConfigParser()

# read_csv_folds
config["READ_CSV"] = {
    "search_header": "Search Query",
    "random_size": 20,
    "csv_location": "..\\ORS_serachData\\search.csv",
    "nrows": 100,
}

# model_evaluation, doccanoJsonl_to_json
config["MODEL_ENTITIES"] = {
    "haircare_entities": "'CONSUMER_TYPE','BRAND', 'ORG', 'CARDINAL', 'DATE', 'GPE', 'LOC', 'MONEY', 'PRODUCT', 'ORDINAL', 'PERCENT', 'NORP', 'EVENT', 'WORK_OF_ART', 'FAC', 'PERSON', 'TIME', 'PRODUCT_DESC', 'QUANTITY', 'HAIR_TYPE', 'QUERY', 'CONSUMER_CATEGORY','FREE_OF','HAIR_CONDITION','HELPS_WITH','INGREDIENT','PRODUCT_BULK','PRODUCT_LINE'"
}

# doccanojsonl_to_json
config["ANNOTATOR_KEYS"] = {"entity_term": "labels", "content_term": "text"}

with open("config_file.ini", "w") as configfile:
    config.write(configfile)
