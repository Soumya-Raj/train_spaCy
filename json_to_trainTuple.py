import json
import time
import sys
import ast

class JsonToSpacy:
    def __init__(self, **kwargs):  # kwargs={"input_fname": sys_arg1}
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

def main(sys_arg1):
    start_time = time.time()
    dict_arg = {"input_fname": sys_arg1}
    convert_json = JsonToSpacy(** dict_arg)
    train_tuples = convert_json.to_spacyformat()
    print(
        f"Converted json to spacy training tuple format in {time.time() - start_time} seconds"
    )

if __name__ == "__main__":
    main(sys.argv[1])
