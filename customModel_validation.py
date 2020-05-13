import spacy
from pathlib import Path
#test_text = "I need a anti breakage sulphate-free leave in conditioning shampoo 10 oz under $30 for curly hair"
test_text = "I need a red dress for a conference in Dubai under $30 from Nordstorm before 30th march midnight"

# test the saved model
print("Loading ner from tained model")
nlp = spacy.load('trained_model_lg3')
# Check the classes have loaded back consistently
ner = nlp.get_pipe("ner")
move_names = list(ner.move_names)
assert nlp.get_pipe("ner").move_names == move_names
doc = nlp(test_text)
print(doc)
for ent in doc.ents:
    print("Validation :",ent.label_, ent.text)