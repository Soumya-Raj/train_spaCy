import spacy
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from config.load_config_file import LoadConfigFile
import time
import sys


class ModelEval:
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.config = LoadConfigFile.read_config_file(self, "config_file.ini")
        #self.entity_config_key = entity_config_key
        self.labels = [self.config["MODEL_ENTITIES"][self.kwargs.get("entity_config_key")]]
        #print(self.labels)

    def model_evaluate(self):
        nlp = spacy.load(self.kwargs.get("model_name"))
        scorer = Scorer()
        validation_set = []
        with open(self.kwargs.get("validation_set"), "w") as f:
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

    def ner_predicte(self):
        test_text = "I need a anti breakage sulphate-free leave in conditioning shampoo 10 oz under $30 for curly hair"
        # test_text = "I need a red dress for a conference in Dubai under $30 from Nordstorm before 30th march midnight"

        # test the saved model
        print("Loading ner from tained model")
        nlp = spacy.load("trained_model_lg3")
        # Check the classes have loaded back consistently
        ner = nlp.get_pipe("ner")
        move_names = list(ner.move_names)
        assert nlp.get_pipe("ner").move_names == move_names
        doc = nlp(test_text)
        print(doc)
        for ent in doc.ents:
            print("Validation :", ent.label_, ent.text)


examples = [
    (
        "Trump says he's answered Mueller's Russia inquiry questions \u2013 live",
        {"entities": [[0, 5, "PERSON"], [25, 32, "PERSON"], [35, 41, "GPE"]]},
    ),
    (
        "Alexander Zverev reaches ATP Finals semis then reminds Lendl who is boss",
        {"entities": [[0, 16, "PERSON"], [55, 60, "PERSON"]]},
    ),
    (
        "Britain's worst landlord to take nine years to pay off string of fines",
        {"entities": [[0, 7, "GPE"]]},
    ),
    (
        "Tom Watson: people's vote more likely given weakness of May's position",
        {"entities": [[0, 10, "PERSON"], [56, 59, "PERSON"]]},
    ),
]


def main(sys_arg1, sys_arg2, sys_arg3):
    start_time = time.time()
    dict_arg = {"model_name": sys_arg1,
                "validation_set": sys_arg2, "entity_config_key": sys_arg3}
    model_eval = ModelEval(**dict_arg)
    results = model_eval.model_evaluate()
    print(results)
    print(
        f"validation executed in {time.time() - start_time} seconds"
    )
    


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
