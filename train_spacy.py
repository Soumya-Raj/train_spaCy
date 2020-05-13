import plac
import random
from pathlib import Path
import spacy
import time
from spacy.util import minibatch, compounding
import ast
import matplotlib.pyplot as plt
import pandas as pd
from spacy.util import decaying
import spacy.scorer

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
TRAIN_DATA = []

with open("doccano_annotated_data\\doccano_fold3_converted.txt", "r") as f:
    TRAIN_DATA = f.read()

TRAIN_DATA = ast.literal_eval(TRAIN_DATA)


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)

# optimisation 1 - batch size
def get_batches(train_data, model_type):
    max_batch_sizes = {"tagger": 32, "parser": 16, "ner": 16, "textcat": 64}
    max_batch_size = max_batch_sizes[model_type]
    if len(train_data) < 1000:
        max_batch_size /= 2
    if len(train_data) < 500:
        max_batch_size /= 2
    batch_size = compounding(1, max_batch_size, 1.001)
    batches = minibatch(train_data, size=batch_size)
    return batches


def main(
    model=None, new_model_name="incremental_model", output_dir="trained_model", n_iter=5
):
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
        # Not current scenario

    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)  # no need to create for en_core_web_sm
    else:
        ner = nlp.get_pipe("ner")
        print("Get ner pipeline")

    # ner.add_label(LABEL)
    for label in LABELS:
        ner.add_label(label)

    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
        # we resume training

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        sizes = compounding(1.0, 4.0, 1.001)
        dropout = decaying(0.6, 0.2, 1e-4)
        # print("DROPOUT {}".format(dropout))
        # batch up the examples using spaCy's minibatch
        loss_plot = []
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            batches = get_batches(TRAIN_DATA, "ner")
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                # print("DROPOUT NEXT {}".format(next(dropout)))
                nlp.update(
                    texts, annotations, sgd=optimizer, drop=next(dropout), losses=losses
                )
            print("Losses", losses)
            loss_plot.append(losses["ner"])
            with open("train_fold.log", "a") as f:
                f.write("%s %s %s\n" % (itn, ",", losses["ner"]))

    # test the trained model
    # test_text = "I need a ORS olive oil for naturals butter creme styling smoothie 12 oz under $100"
    test_text = "I need a anti breakage sulphate-free leave in conditioning shampoo 10 oz under $30 for curly hair"
    doc = nlp(test_text)

    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output dir
    with nlp.use_params(optimizer.averages):
        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
            nlp.meta["name"] = new_model_name  # rename model
            nlp.to_disk(output_dir)
            print("Saved model to", output_dir)

    # test the saved model
    print("Loading from", Path.cwd())
    nlp = spacy.load(output_dir)
    # Check the classes have loaded back consistently
    ner = nlp.get_pipe("ner")
    move_names = list(ner.move_names)
    assert nlp.get_pipe("ner").move_names == move_names
    doc = nlp(test_text)
    print(doc)
    for ent in doc.ents:
        print("Validation:", ent.label_, ent.text)

    # plot training loss
    plt.plot(loss_plot)
    plt.show()


start_time = time.time()
if __name__ == "__main__":
    plac.call(main)
# ('en_core_web_sm', Path.cwd(), 100)
print("--- Total training time : %s seconds ---" % (time.time() - start_time))
