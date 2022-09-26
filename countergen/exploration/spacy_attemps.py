#%%
import spacy

nlp = spacy.load("en_core_web_sm")
# %%
doc = nlp("He's angry. Sally is not, she is happy, because her Mom died.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
# %%
doc2 = nlp("Apple is looking at buying U.K. startup for $1 billion")
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
# %%
import nltk
from nltk.corpus import names

nltk.download("names")
# Read the names from the files.
# Label each name with the corresponding gender.
male_names = [f'"{name}",' for name in names.words("female.txt")]
print(" ".join(male_names))
# female_names = [name for name in names.words("female.txt")]

# %%
