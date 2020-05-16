import logging
import os
import configparser
import pandas as pd
import spacy
import time
import json
from io import StringIO
import random
from pandas import json_normalize
import sys

#sys.path.append("..")
from config.load_config_file import LoadConfigFile


class ReadCSV:
    def __init__(self, **kwargs):
        # kwargs=dict_arg = {"model_name": sys_arg1, "output_fname": sys_arg2,"start_index": sys_arg3, "end_index": sys_arg4}
        self.kwargs = kwargs
        config = LoadConfigFile("config/config_file.ini").read_config_file()
        self.speech_header = config["READ_CSV"]["search_header"]
        self.n_rows = config["READ_CSV"]["nrows"]
        self.csv_fname = os.path.relpath(config["READ_CSV"]["csv_location"])
        print(self.csv_fname)

    def auto_annotate_data(self, df_dict):
        nlp = spacy.load(os.path.relpath(self.kwargs.get("model_name")))
        annotated_list = []
        for d in df_dict:
            doc = nlp(str(d[self.speech_header]))
            for sent in doc.sents:
                labels = []
                for e in sent.ents:
                    labels.append([e.start_char, e.end_char, e.label_])
                annotated_list.append({"text": sent.text, "labels": labels})
        annotated_json = json.dumps(annotated_list)
        #print(annotated_json)
        return annotated_json

    def randomize_list(self, df_list):
        random_size = 20
        random_list = random.sample(range(len(df_list)), random_size)
        length = len(random_list)
        middle_index = length // 2
        first_half = random_list[:middle_index]
        second_half = random_list[middle_index:]
        for i in first_half:
            df_list[i][self.speech_header] = (
                "I need a " + df_list[i][self.speech_header]
            )
        for i in second_half:
            df_list[i][self.speech_header] = (
                df_list[i][self.speech_header] + " under $30"
            )

    def read_csv_as_df(self):
        fields = [self.speech_header]

        df = pd.read_csv(
            self.csv_fname,
            usecols=fields,
            nrows=int(self.n_rows),
            skiprows=range(
                int(self.kwargs.get("start_index")), int(self.kwargs.get("end_index"))
            ),
        )
        df.info()
        df_dict = df.to_dict("r")
        return df_dict

    def json_to_jsonl(self, d_json):
        output_path = os.path.relpath("../data/auto_annotated_data")
        json_converted = json.loads(d_json)
        # convert json dump to jsonl format of Doccano
        df_annotated = pd.DataFrame.from_dict(
            json_normalize(json_converted), orient="columns"
        )
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        df_annotated.to_json(
                os.path.relpath(f"../data/auto_annotated_data/{self.kwargs.get('output_fname')}") ,
            orient="records",
            lines=True,
        )


# def main(sys_arg1, sys_arg2, sys_arg3, sys_arg4):
#     start_time = time.time()
#     dict_arg = {
#         "model_name": sys_arg1,
#         "output_fname": sys_arg2,
#         "start_index": sys_arg3,
#         "end_index": sys_arg4,
#     }
#     read_csv = ReadCSV(**dict_arg)
#     df_dict = read_csv.read_csv_as_df()
#     # read_csv.randomize_list(df_dict)
#     d_json = read_csv.auto_annotate_data(df_dict)
#     read_csv.json_to_jsonl(d_json)

#     print(
#         f"Auto annotated training data using {sys_arg1} in {time.time() - start_time} seconds"
#     )


# if __name__ == "__main__":
#     main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
# # command line format : training_model output_jsonl_filename start_index_skip_row end_index_skip_row
