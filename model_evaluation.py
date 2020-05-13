import spacy
from spacy.gold import GoldParse
from spacy.scorer import Scorer

LABELS = [
    "BRAND",
    "ORG",
    "CARDINAL",
    "DATE",
    "GPE",
    "LOC",
    "MONEY",
    "PRODUCT",
    "ORDINAL",
    "PERCENT",
    "NORP",
    "EVENT",
    "WORK_OF_ART",
    "FAC",
    "PERSON",
    "TIME",
    "PRODUCT_DESC",
    "QUANTITY",
    "HAIR_TYPE",
    "QUERY",
    "CONSUMER_TYPE",
]


def custom_model_evaluate(nlp, examples, label=LABELS):
    scorer = Scorer()
    for input_, annot in examples:
        text_entities = []
        for entity in annot.get("entities"):
            # print(entity)
            for l in label:
                print(l)
                if l in entity:
                    text_entities.append(entity)
        doc_gold_text = nlp.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=text_entities)
        pred_value = nlp(input_)
        scorer.score(pred_value, gold)
    return scorer.scores


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

nlp = spacy.load("en_core_web_sm")
results = custom_model_evaluate(nlp, examples)
print(results)
