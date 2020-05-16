import json
import os
import ast

from config.load_config_file import LoadConfigFile


class ConvertToJson:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.config = LoadConfigFile("config/config_file.ini").read_config_file()

    def jsonl_to_json(self):
        fo = open(os.path.relpath(self.kwargs.get("input_fname")), "r")

        lines = fo.readlines()

        final_json = []
        entity_term = self.config["ANNOTATOR_KEYS"]["entity_term"]
        content_term = self.config["ANNOTATOR_KEYS"]["content_term"]

        for line in lines:
            line = json.loads(line)
            if entity_term in line:
                line["entities"] = line.pop(entity_term)
            else:
                line["entities"] = []

            tmp_ents = []
            for e in line["entities"]:
                if e[2] in ast.literal_eval(
                    self.config["MODEL_ENTITIES"]["haircare_entities"]
                ):
                    tmp_ents.append([e[0], e[1], e[2]])
                line["entities"] = tmp_ents
            final_json.append(
                {"content": line[content_term], "entities": line["entities"]}
            )

        output_fpath = os.path.relpath("../data/doccano_annotated_data")
        if not os.path.exists(output_fpath):
            os.mkdir(output_fpath)
        with open(
            os.path.relpath(
                f"../data/doccano_annotated_data/{self.kwargs.get('output_fname')}"
            ),
            "w",
        ) as f:
            json.dump(final_json, f)
