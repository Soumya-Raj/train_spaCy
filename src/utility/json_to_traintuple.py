import json


class JsonToSpacy:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def to_spacyformat(self):
        input_fname = self.kwargs.get("input_fname")
        with open(input_fname) as f:
            train_data = json.load(f)

        train_tuples = []
        for data in train_data:
            ents = [tuple(entity) for entity in data["entities"]]
            train_tuples.append((data["content"], {"entities": ents}))

        with open("{}".format(input_fname.replace("json", "txt")), "w") as write:
            write.write(str(train_tuples))
        return train_tuples
