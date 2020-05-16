import spacy
from spacy.gold import GoldParse
from spacy.scorer import Scorer
import sys
sys.path.append('..')
import os
from config.load_config_file import LoadConfigFile
import time
import ast
import argparse


class ModelEval:

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        config = LoadConfigFile("config/config_file.ini").read_config_file()
        #print(config.sections())
        self.labels = [config["MODEL_ENTITIES"][self.kwargs.get("entity_config_key")]]

    def model_evaluate(self):
        nlp = spacy.load(os.path.realpath(self.kwargs.get("model_path")))
        print("Loaded %s" % self.kwargs.get("model_path"))
        scorer = Scorer()
        validation_set = self.kwargs.get("validation_set")
        print(validation_set)
        # with open(self.kwargs.get("validation_set"), "w") as f:
        #     validation_set = f.read()
        for input_, annot in validation_set:
            text_entities = []
            for entity in annot.get("entities"):
                print(self.labels)
                for l in self.labels:
                    if l in entity:
                        text_entities.append(entity)
            doc_gold_text = nlp.make_doc(input_)
            gold = GoldParse(doc_gold_text, entities=text_entities)
            pred_value = nlp(input_)
            scorer.score(pred_value, gold)
        return scorer.scores

    def ner_predicte(self):
        test_text = "I need a anti breakage sulphate-free leave in conditioning shampoo 10 oz under $30 for curly hair"
        print("Loading ner from tained model")
        nlp = spacy.load("trained_model_lg3")
        ner = nlp.get_pipe("ner")
        move_names = list(ner.move_names)
        assert nlp.get_pipe("ner").move_names == move_names
        doc = nlp(test_text)
        print(doc)
        for ent in doc.ents:
            print("Validation :", ent.label_, ent.text)


def main(args):
    with open("validation_converted.txt") as f:
        EXAMPLES = f.read()
        #print(EXAMPLES)
    start_time = time.time()
    dict_arg = {"model_path": args.model_path,
                "validation_set": EXAMPLES, "entity_config_key": args.entity_key}
    model_eval = ModelEval(**dict_arg)
    results = model_eval.model_evaluate()
    print(results)
    print(
        f"validation executed in {time.time() - start_time} seconds"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, required=True, help="Folder path of trained model")
    parser.add_argument('--validation_set', type = str, required=True, help="Folder path of validation set file")
    parser.add_argument('--entity_key', type = str, required=True, help="Folder path of trained model")
    
    args = parser.parse_args()

    main(args)
