import torch
import spacy

# spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")
doc = nlp("R.LUKAKU 48 70")
print([(ent.text, ent.label_) for ent in doc.ents])
