import json
from collections import defaultdict
from pathlib import Path
from random import choice
from typing import Callable, DefaultDict, Dict, Iterable, List, Mapping, OrderedDict, Sequence, Tuple

import re
from attrs import define

from countergen.types import Category, Augmenter, Input
from countergen.tools.utils import other
from countergen.config import MODULE_PATH

DEFAULT_CONVERTERS_PATHS: Mapping[str, str] = {
    "gender": f"{MODULE_PATH}/data/converters/gender.json",
    "west_v_asia": f"{MODULE_PATH}/data/converters/west_v_asia.json",
}


@define
class ConversionDataset:
    categories: Tuple[Category, Category]
    correspondances: List[Tuple[List[str], List[str]]]

    @classmethod
    def from_json(cls, json_dict: OrderedDict):
        categories = tuple(json_dict["categories"])
        correspondances_maps = json_dict["correspondances"]
        cat_0, cat_1 = categories
        correspondances = []
        for m in correspondances_maps:
            correspondances.append((m[cat_0], m[cat_1]))
        return ConversionDataset(categories, correspondances)  # type: ignore


Transformation = Callable[[str], str]
CorrespondanceDict = Dict[Category, DefaultDict[str, List[str]]]
DEFAULT_TRANSFORMATIONS = [
    lambda s: s.lower(),
    lambda s: s.upper(),
    lambda s: s.capitalize(),
]


@define
class SimpleAugmenter(Augmenter):
    categories: Tuple[Category, Category]
    correspondance_dict: CorrespondanceDict
    word_regex: str = r"([A-Za-zÀ-ÖØ-öø-ÿ]+\-[A-Za-zÀ-ÖØ-öø-ÿ]+)|[A-Za-zÀ-ÖØ-öø-ÿ]+"

    @classmethod
    def from_default(cls, name: str = "gender"):
        return SimpleAugmenter.from_json(DEFAULT_CONVERTERS_PATHS[name])

    @classmethod
    def from_json(
        cls,
        path: str,
        transformations: Iterable[Transformation] = DEFAULT_TRANSFORMATIONS,
    ):
        with Path(path).open("r", encoding="utf-8") as f:
            json_dict = json.loads(f.read())
            return SimpleAugmenter.from_ds(ConversionDataset.from_json(json_dict), transformations)

    @classmethod
    def from_ds(
        cls,
        ds: ConversionDataset,
        transformations: Iterable[Transformation] = DEFAULT_TRANSFORMATIONS,
    ):
        correspondance_dict: CorrespondanceDict = {}
        for c in ds.categories:
            correspondance_dict[c] = defaultdict(lambda: [])

        for correspondance in ds.correspondances:
            correspondance_t = {
                c: {t.__code__: list(map(t, l)) for t in transformations} for c, l in zip(ds.categories, correspondance)
            }

            for c, l in zip(ds.categories, correspondance):
                for word in l:
                    for t in transformations:
                        correspondance_dict[c][t(word)] += correspondance_t[other(ds.categories, c)][t.__code__]

        return SimpleAugmenter(ds.categories, correspondance_dict)

    def transform(self, inp: Input, to: Category) -> Input:
        from_category = other(self.categories, to)
        p = re.compile(self.word_regex)
        new_inp_pieces = []
        end = 0
        for group in p.finditer(inp):
            word = inp[group.start() : group.end()]
            if self.correspondance_dict[from_category][word]:
                new_inp_pieces.append(inp[end : group.start()])
                new_inp_pieces.append(choice(self.correspondance_dict[from_category][word]))
                end = group.end()

        new_inp_pieces.append(inp[end:])
        return "".join(new_inp_pieces)
