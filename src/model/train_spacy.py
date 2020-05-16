import random
from pathlib import Path
import spacy
import time
from spacy.util import minibatch, compounding
import ast
import matplotlib.pyplot as plt
from spacy.util import decaying
import spacy.scorer
from config.load_config_file import LoadConfigFile
from utility.json_to_traintuple import JsonToSpacy
import sys

import os


class TrainSpacy:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.config = LoadConfigFile("config/config_file.ini").read_config_file()

    # optimisation 1 - batch size
    def get_batches(self, train_data, model_type):
        max_batch_sizes = {"tagger": 32, "parser": 16, "ner": 16, "textcat": 64}
        max_batch_size = max_batch_sizes[model_type]
        if len(train_data) < 1000:
            max_batch_size /= 2
        if len(train_data) < 500:
            max_batch_size /= 2
        batch_size = compounding(1, max_batch_size, 1.5)
        batches = minibatch(train_data, size=batch_size)
        return batches

    def train_spacymodel(self):
        train_data = JsonToSpacy.to_spacyformat(self)
        # print(train_data)

        labels = ast.literal_eval(
            self.config["MODEL_ENTITIES"][self.kwargs.get("entity_config_key")]
        )
        # print(labels)
        if self.kwargs.get("model") is not None:
            nlp = spacy.load(self.kwargs.get("model"))
            print("Loaded model '%s'" % self.kwargs.get("model"))
        else:
            nlp = spacy.blank("en")
            print("Created blank 'en' model")

        if "ner" not in nlp.pipe_names:
            ner = nlp.create_pipe("ner")
            nlp.add_pipe(ner)
            print("Ner pipeline added")
        else:
            ner = nlp.get_pipe("ner")
            print("Get ner pipeline")

        for label in labels:
            ner.add_label(label)

        if self.kwargs.get("model") is None:
            optimizer = nlp.begin_training()
        else:
            optimizer = nlp.resume_training()

        pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
        with nlp.disable_pipes(*other_pipes):
            # optimisation 2 - dropout
            dropout = decaying(0.6, 0.2, 1e-4)
            # print("DROPOUT {}".format(dropout))
            loss_plot = []
            for itn in range(int(self.kwargs.get("n_iter"))):
                random.shuffle(train_data)
                batches = self.get_batches(train_data, "ner")
                losses = {}
                for batch in batches:
                    texts, annotations = zip(*batch)
                    # print("DROPOUT NEXT {}".format(next(dropout)))
                    nlp.update(
                        texts,
                        annotations,
                        sgd=optimizer,
                        drop=next(dropout),
                        losses=losses,
                    )
                print("Losses", losses)
                loss_plot.append(losses["ner"])
                with open(
                    os.path.relpath("../reports/log/training_loss.log"), "a"
                ) as f:
                    f.write("Epoch %s loss: %s\n" % (itn, losses["ner"]))

        # test trained model
        test_text = "I need a anti breakage sulphate-free leave in conditioning shampoo 10 oz under $30 for curly hair"
        doc = nlp(test_text)

        print("Entities in '%s'" % test_text)
        for ent in doc.ents:
            print(ent.label_, ent.text)

        # save model to output_dir
        output_dir = self.kwargs.get("output_dir")
        with nlp.use_params(optimizer.averages):
            if output_dir is not None:
                output_dir = Path(output_dir)
                if not output_dir.exists():
                    output_dir.mkdir()
                nlp.meta["name"] = self.kwargs.get("new_model_name")
                nlp.to_disk(output_dir)
                print("Saved model to", output_dir)

        # test saved model
        print("Loading from", Path.cwd())
        nlp = spacy.load(output_dir)
        ner = nlp.get_pipe("ner")
        move_names = list(ner.move_names)
        assert nlp.get_pipe("ner").move_names == move_names
        doc = nlp(test_text)
        for ent in doc.ents:
            print("Testing saved model:", ent.label_, ent.text)

        # plot and save training loss
        plt.plot(loss_plot)
        plt.savefig(
            os.path.realpath(
                f"../reports/figures/{self.kwargs.get('new_model_name')}.png"
            )
        )
