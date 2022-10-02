#%%
import json
from pathlib import Path

load_path = "../my_data/stereotypes.json"
save_path = "../countergen/data/datasets/gender-stereotypes.jsonl"

data = json.load(Path(load_path).open())
import re

word_regex: str = r"([A-Za-zÀ-ÖØ-öø-ÿ]+\-[A-Za-zÀ-ÖØ-öø-ÿ]+)|[A-Za-zÀ-ÖØ-öø-ÿ]+"
p = re.compile(word_regex)
gendered_words = {"he", "she", "her", "him", "his", "women", "woman", "men", "man"}

sentences = data["data"]["intersentence"]

gender_stereotypes = []

for s in sentences:
    if s["bias_type"] != "gender":
        continue
    context = s["context"]
    outs = []
    for out in s["sentences"]:
        nb_sterotype_votes = len([x for x in out["labels"] if x["label"] == "stereotype"])
        continuation = out["sentence"]
        words = set([continuation[group.start() : group.end()].lower() for group in p.finditer(continuation)])
        # print(words)
        if len(words.intersection(gendered_words)) != 0:
            continue
        if nb_sterotype_votes > len(out["labels"]) / 2:
            outs.append(out["sentence"])
    if outs:
        gender_stereotypes.append((context, outs))
print(gender_stereotypes)
print(len(gender_stereotypes))

lines_written = 0
with open(save_path, "w", encoding="utf-8") as outfile:
    for inp, outs in gender_stereotypes:
        json_dict = {"input": inp, "outputs": [" "+ o for o in outs]}
        json.dump(json_dict, outfile)
        outfile.write("\n")
        lines_written += 1
print(f"Done! {lines_written} lines written.")

# %%
