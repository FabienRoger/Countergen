import json
from ast import Or
from pathlib import Path
from typing import Any, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Union, cast

import cattrs
import countergen.config
from attrs import define
from countergen.augmentation.simple_augmenter import SimpleAugmenter
from countergen.config import MODULE_PATH
from countergen.tools.utils import FromAndToJson, all_same, maybe_tqdm
from countergen.types import AugmentedSample, Augmenter, Category, Input, Outputs, Paraphraser, Sample, Variation

DEFAULT_DS_PATHS: Mapping[str, str] = {
    "doublebind-negative-1token": f"{MODULE_PATH}/data/datasets/doublebind-negative-1token.jsonl",
    "doublebind-negative": f"{MODULE_PATH}/data/datasets/doublebind-negative.jsonl",
    "doublebind-positive-1token": f"{MODULE_PATH}/data/datasets/doublebind-positive-1token.jsonl",
    "doublebind-positive": f"{MODULE_PATH}/data/datasets/doublebind-positive.jsonl",
    "doublebind-likable": f"{MODULE_PATH}/data/datasets/doublebind-likable.jsonl",
    "doublebind-unlikable": f"{MODULE_PATH}/data/datasets/doublebind-unlikable.jsonl",
    "male-stereotypes": f"{MODULE_PATH}/data/datasets/male-stereotypes.jsonl",
    "female-stereotypes": f"{MODULE_PATH}/data/datasets/female-stereotypes.jsonl",
    "tiny-test": f"{MODULE_PATH}/data/datasets/tiny-test.jsonl",
    "twitter-sentiment": f"{MODULE_PATH}/data/datasets/twitter-sentiment.jsonl",
    "hate": f"{MODULE_PATH}/data/datasets/hate-test.jsonl",
}

DEFAULT_AUGMENTED_DS_PATHS: Mapping[str, str] = {
    "doublebind-positive-paraphrased": f"{MODULE_PATH}/data/augdatasets/doublebind-positive-paraphrased.jsonl",
    "doublebind-negative-paraphrased": f"{MODULE_PATH}/data/augdatasets/doublebind-negative-paraphrased.jsonl",
    "doublebind-likable-paraphrased": f"{MODULE_PATH}/data/augdatasets/doublebind-likable-paraphrased.jsonl",
    "doublebind-unlikable-paraphrased": f"{MODULE_PATH}/data/augdatasets/doublebind-unlikable-paraphrased.jsonl",
    "male-stereotypes": f"{MODULE_PATH}/data/augdatasets/male-stereotypes.jsonl",
    "female-stereotypes": f"{MODULE_PATH}/data/augdatasets/female-stereotypes.jsonl",
    "tiny-test-aug-gender": f"{MODULE_PATH}/data/augdatasets/tiny-test.jsonl",
    "twitter-sentiment-aug-gender": f"{MODULE_PATH}/data/augdatasets/twitter-sentiment.jsonl",
    "hate-aug-muslim": f"{MODULE_PATH}/data/augdatasets/hate-test-muslims.jsonl",
    "bias-from-probs": f"{MODULE_PATH}/data/augdatasets/bias-from-probs.jsonl",
    "simple-gender": f"{MODULE_PATH}/data/augdatasets/simple-gender.jsonl",
}


@define
class SimpleAugmentedSample(Sample, AugmentedSample):
    """AugmentedSample which explicitly stores all variations."""

    variations: List[Variation] = []

    def get_variations(self) -> Sequence[Variation]:
        return self.variations

    def get_outputs(self) -> Outputs:
        return self.outputs

    @classmethod
    def from_sample(cls, s: Sample, variations: List[Variation] = []) -> "SimpleAugmentedSample":
        return SimpleAugmentedSample(s.input, s.outputs, variations)


@define
class Dataset:
    samples: List[Sample]

    @classmethod
    def from_default(cls, name: str) -> "Dataset":
        """Load one of the defaults datasets from "DEFAULT_DS_PATHS"."""
        if name not in DEFAULT_DS_PATHS:
            raise ValueError(
                f"""Default name '{name}' is not a default dataset. Choose one in {list(DEFAULT_DS_PATHS.keys())}"""
            )
        return Dataset.from_jsonl(DEFAULT_DS_PATHS[name])

    @classmethod
    def from_jsonl(cls, path: str) -> "Dataset":
        """Load a dataset from a jsonl file.

        Expected format of each line:
        {"input": "<INP>", "outputs": ["<OUT1>", "<OUT2>", ...]}
        where you have at least one accepted output per input."""
        with Path(path).open("r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
            samples = []
            for d in data:
                samples.append(Sample.from_json_dict(d))
        return Dataset(samples)

    def augment(self, converters: Iterable[Augmenter]) -> "AugmentedDataset":
        """Use the augmenters to generate variations of this dataset."""
        return generate_all_variations(converters, self)


@define
class AugmentedDataset:
    samples: List[SimpleAugmentedSample]

    @classmethod
    def from_default(cls, name: str) -> "AugmentedDataset":
        """Load one of the defaults datasets from "DEFAULT_AUGMENTED_DS_PATHS"."""
        if name not in DEFAULT_AUGMENTED_DS_PATHS:
            raise ValueError(
                f"Default name '{name}' is not a default augmented dataset. Choose one in {list(DEFAULT_AUGMENTED_DS_PATHS.keys())}"
            )
        return AugmentedDataset.from_jsonl(DEFAULT_AUGMENTED_DS_PATHS[name])

    @classmethod
    def from_jsonl(cls, path: str) -> "AugmentedDataset":
        with Path(path).open("r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
            samples = []
            for d in data:
                samples.append(SimpleAugmentedSample.from_json_dict(d))
        return AugmentedDataset(samples)

    def save_to_jsonl(self, path: str):
        with Path(path).open("w", encoding="utf-8") as f:
            for sample in self.samples:
                json.dump(sample.to_json_dict(), f)
                f.write("\n")

    def to_unbiased_dataset(self) -> Dataset:
        """Make each variation a new input."""
        new_samples: List[Sample] = []

        for aug_sample in self.samples:
            for variation in aug_sample.get_variations():
                new_samples.append(Sample(input=variation.text, outputs=aug_sample.outputs))

        return Dataset(new_samples)


def generate_variations(variation: Variation, augmenter: Augmenter) -> List[Variation]:
    if isinstance(augmenter, Paraphraser):
        return generate_paraphrase(variation, cast(Paraphraser, augmenter))

    new_variations = [
        Variation(augmenter.transform(variation.text, category), variation.categories + (category,))
        for category in augmenter.categories
    ]
    if not all_same([v.text for v in new_variations]):
        return new_variations
    else:
        return [variation]


def generate_paraphrase(variation: Variation, augmenter: Paraphraser) -> List[Variation]:
    new_text = augmenter.transform(variation.text)
    if new_text == variation.text:
        return [variation]
    else:
        return [variation, Variation(new_text, variation.categories)]


def generate_all_variations(augmenters: Iterable[Augmenter], ds: Dataset) -> AugmentedDataset:
    """Apply each augmenter to each sample of the dataset for each available target category.

    It first replaces samples with the transformed samples through the first augmenter,
    then replaces these with samples transformed through the second augmenter, and so on.
    Return an number of variations that can be exponential in the number of augmenters.

    Remove duplicates.

    If an augmenter is a paraphrase, keep the original input too."""

    augmented_samples = []

    for sample in maybe_tqdm(ds.samples, countergen.config.verbose >= 2):
        variations = [Variation(sample.input, ())]
        for augmenter in augmenters:
            new_variations = []
            for v in variations:
                new_variations += generate_variations(v, augmenter)
            variations = new_variations
        augmented_samples.append(SimpleAugmentedSample.from_sample(sample, variations))
    return AugmentedDataset(augmented_samples)
