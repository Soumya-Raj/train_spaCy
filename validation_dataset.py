import pandas as pd
import spacy
import time
import json
from io import StringIO
import random
from pandas.io.json import json_normalize

def text_to_doccano(df_dict):
    nlp = spacy.load('trained_model')
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
    #doccano_json = StringIO(doccano_json)
    return doccano_json

def randomize_list(df_list):
    random_size = 20
    random_list = random.sample(range(len(df_list)), random_size)
    length = len(random_list)
    middle_index = length//2
    first_half = random_list[:middle_index]
    second_half = random_list[middle_index:]
    print(first_half)
    print(second_half)
    for i in first_half:
        df_list[i]['Search Query'] = "I need a " + df_list[i]['Search Query'] + "for curly hair"
    for i in second_half:
        df_list[i]['Search Query'] = df_list[i]['Search Query'] + " 12 fl.oz above $20"

fields = ['Search Query']
df = pd.read_csv('ORS_serachData\\search.csv', usecols=fields, nrows=100, skiprows=range(1,7000))
df.info()
df_dict = df.to_dict("r")

start = time.time()
randomize_list(df_dict)
d_json = text_to_doccano(df_dict)
d_json_new=json.loads(d_json)
# with open('ORS_doccano_data.json', 'w') as f:
#     json.dump(d_json, f)

df_new = pd.DataFrame.from_dict(json_normalize(d_json_new), orient='columns')

df_new.head()
df_new.to_json('auto_annotated_data\\validation_set',orient="records", lines=True)
   
print("Created validation set from dataset in {}".format(time.time() - start))


