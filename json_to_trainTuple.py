import json

filename = input("Enter file: ")
print(filename)


with open(filename) as train_data:
    train = json.load(train_data)

TRAIN_DATA = []
for data in train:
    ents = [tuple(entity) for entity in data["entities"]]
    TRAIN_DATA.append((data["content"], {"entities": ents}))


with open("{}".format(filename.replace("json", "txt")), "w") as write:
    write.write(str(TRAIN_DATA))

print("Converted json to spacy train data format - list of tuples")
