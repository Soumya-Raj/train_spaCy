import pandas as pd
import spacy
import time
import json
from io import StringIO
import random
from pandas import json_normalize
import sys
import configparser
import os 
import logging

def auto_annotate_data(df_dict,model_name,output_fname,start_index,end_index):
    nlp = spacy.load(model_name)
    doccano_list = list()
    for d in df_dict:
        d['Search Query'] = str(d['Search Query'])
        doc = nlp(d['Search Query'] )
        for sent in doc.sents:
            labels = list()
            for e in sent.ents:
                labels.append([e.start_char, e.end_char, e.label_])
            doccano_list.append({'text': sent.text, "labels": labels})
    doccano_json = json.dumps(doccano_list)
    #print(doccano_json)
    return doccano_json

def randomize_list(df_list):
    random_size = 20
    random_list = random.sample(range(len(df_list)), random_size)
    length = len(random_list)
    middle_index = length//2
    first_half = random_list[:middle_index]
    second_half = random_list[middle_index:]
    for i in first_half:
        df_list[i]['Search Query'] = "I need a " + df_list[i]['Search Query']
    for i in second_half:
        df_list[i]['Search Query'] = df_list[i]['Search Query'] + " under $30"

config = configparser.ConfigParser()
config.read('config\\config_file.ini', encoding='utf-8')

fields = [config['READ_CSV']['search_header']]
df = pd.read_csv(config['READ_CSV']['csv_location'], usecols=fields, nrows=int(config['READ_CSV']['nrows']), skiprows=range(int(sys.argv[3]),int(sys.argv[4])))
df.info()
df_dict = df.to_dict("r")

start_time = time.time()
randomize_list(df_dict)
d_json = auto_annotate_data(df_dict,sys.argv[1])
d_json_convert=json.loads(d_json)

#convert json dump to jsonl format of Doccano
df_annotated = pd.DataFrame.from_dict(json_normalize(d_json_convert), orient='columns')
if not os.path.exists('auto_annotated_data'):
    os.mkdir('auto_annotated_data')
df_annotated.to_json('auto_annotated_data\\'+sys.argv[2],orient="records", lines=True)
   
print("Auto annotated training data using {} in {} seconds".format(sys.argv[1], time.time() - start_time))

#command line format : training_model output_jsonl_filename start_index_skip_row end_index_skip_row


