import sys
import json


filename = input("Enter file: ")
print(filename)
fo = open(filename, "r")

lines = fo.readlines()

final_json=[]
#final_json.append('[')
for line in lines:
    line =json.loads(line)
    if "labels" in line:
    	line["entities"] = line.pop("labels")
    else:
	    line["entities"] = []

    tmp_ents = []
    for e in line["entities"]:
        if e[2] in ['BRAND', 'ORG', 'CARDINAL', 'DATE', 'GPE', 'LOC', 'MONEY', 'PRODUCT', 'ORDINAL', 'PERCENT', 'NORP', 'EVENT', 'WORK_OF_ART', 'FAC', 'PERSON', 'TIME', 'PRODUCT_DESC', 'QUANTITY', 'HAIR_TYPE', 'QUERY', 'CONSUMER_TYPE']:
     	    tmp_ents.append([e[0], e[1], e[2]])
        line["entities"] = tmp_ents
    final_json.append({"content": line["text"], "entities": line["entities"] })
    #final_json.append(',')
#final_json.append(']')
with open('doccano_annotated_data\\doccano_fold3_converted.json', 'w') as f:
    json.dump(final_json, f)
print("Converted doccano json to spacy required json")

