# %%
import json

with open("../countergen/data/examples/doublebind.jsonl") as f:
    data = [json.loads(line) for line in f]
#%%
from collections import defaultdict

transformations = [
    lambda s: s.lower(),
    lambda s: s.upper(),
    lambda s: s.capitalize(),
]

correspondance_dict = {}
with open("../countergen/data/converters/gender.json") as f:
    corres_data = json.loads(f.read())
    categories = corres_data["categories"]
    assert len(categories) == 2
    for c in categories:
        correspondance_dict[c] = defaultdict(lambda: [])

    def other_category(c):
        return [cat for cat in categories if cat != c][0]

    for correspondance in corres_data["correspondances"]:
        correspondance_t = {
            c: {t.__code__: list(map(t, l)) for t in transformations} for c, l in correspondance.items()
        }

        for c, l in correspondance.items():
            for word in l:
                for t in transformations:
                    correspondance_dict[c][t(word)] += correspondance_t[other_category(c)][t.__code__]

# %%
import spacy
from random import choice

nlp = spacy.load("en_core_web_sm")


def detect(doc, category):
    return any(len(correspondance_dict[category][t.text]) > 0 for t in doc)


def transform(doc, from_category):
    """Swap the cateogry"""
    r = ""
    position_in_text = 0
    for t in doc:
        if correspondance_dict[from_category][t.text]:
            r += doc.text[position_in_text : t.idx] + choice(correspondance_dict[from_category][t.text])
            position_in_text = t.idx + len(t)
    return r + doc.text[position_in_text:]


for d in data:
    inp = d["input"]
    print(inp)
    doc = nlp(inp)
    # print(
    #     " ".join(
    #         f"{t.text}-{t.idx}-{t.text in correspondance_dict['male']}-{t.text in correspondance_dict['female']}"
    #         for t in doc
    #     )
    # )
    # print(detect(doc, "female"), detect(doc, "male"))
    print(transform(doc, "female"), transform(doc, "male"))

# %%
