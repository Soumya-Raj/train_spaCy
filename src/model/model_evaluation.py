import spacy
from spacy.gold import GoldParse
from spacy.scorer import Scorer

import os
from config.load_config_file import LoadConfigFile


class ModelEval:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        config = LoadConfigFile("config/config_file.ini").read_config_file()
        self.labels = [config["MODEL_ENTITIES"][self.kwargs.get("entity_config_key")]]

    #in progress
    def model_evaluate(self):
        nlp = spacy.load(os.path.realpath(self.kwargs.get("model_path")))
        print("Loaded %s" % self.kwargs.get("model_path"))
        scorer = Scorer()
        with open(os.path.relpath(self.kwargs.get("validation_set")), "r") as f:
            validation_set = f.read()
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

    def ner_predict(self):
        test_text = "I need a anti breakage sulphate-free leave in conditioning shampoo 10 oz under $30 for curly hair"
        print("Loading ner from tained model")
        nlp = spacy.load(self.kwargs.get("model_path"))
        ner = nlp.get_pipe("ner")
        move_names = list(ner.move_names)
        assert nlp.get_pipe("ner").move_names == move_names
        doc = nlp(test_text)
        print(doc)
        for ent in doc.ents:
            print("NER prediction on trained model :", ent.label_, ent.text)
