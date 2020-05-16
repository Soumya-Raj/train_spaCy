# train_spaCy
Incrementally train a pre-trained or custom machine learning model for NER using python based NLP library- spaCy and text annotation tool - Doccano.

## Getting Started
Clone/Download the project using https link:
```
git clone https://github.com/Soumya-Raj/train_spaCy.git
```

## Initialise and Run
1. Install requisites using requirements.txt
2. src/controller.py acts as main  
3. Run required service using command line arguments  

## Steps to incrementally train a model 
1. Use read_csv_folds to generate partially annotated fold of training data using any of the pretrained spacy models
2. Manually annotate rest of the entities using Docanno - https://github.com/doccano/doccano 
3. Use utility functions to convert doccano annotated data to spacy's training data format
4. Use train_spacy to train the fold of data and visualize loss log and loss plot in reports 
5. Repeat the same for whole dataset to train the model incrementally 

