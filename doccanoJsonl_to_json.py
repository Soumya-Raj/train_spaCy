import sys
import json
import os
import configparser
import time

def jsonl_to_json(input_fname, output_fname):
    fo = open(input_fname, "r")

    lines = fo.readlines()

    final_json=[]
    config = configparser.ConfigParser()
    config.read('config\\config_file.ini', encoding='utf-8')
    entity_term=config['ANNOTATOR_KEYS']['entity_term']
    content_term = config['ANNOTATOR_KEYS']['content_term']

    for line in lines:
        line =json.loads(line)
        if entity_term in line:
            line["entities"] = line.pop(entity_term)
        else:
            line["entities"] = []

        tmp_ents = []
        for e in line["entities"]:
            if e[2] in [config['MODEL_ENTITIES']['haircare_entities']]:
                tmp_ents.append([e[0], e[1], e[2]])
            line["entities"] = tmp_ents
        final_json.append({"content": line[content_term], "entities": line["entities"] })

    if not os.path.exists('doccano_annotated_data'):
        os.mkdir('doccano_annotated_data')
    with open('doccano_annotated_data\\'+output_fname, 'w') as f:
        json.dump(final_json, f)

start_time = time.time()    
jsonl_to_json(sys.argv[1], sys.argv[2])
print("Converted doccano json to spacy required json in {} seconds".format(time.time() - start_time))

