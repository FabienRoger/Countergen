from typing import List, Optional, Union

import fire  # type: ignore

from countergen import (
    DEFAULT_CONVERTERS_PATHS,
    AugmentedDataset,
    Dataset,
    SimpleAugmenter,
    api_to_generative_model,
    evaluate_and_print,
    get_generative_model_evaluator,
)
from countergen.augmentation.paraphraser import LlmParaphraser
from countergen.types import Augmenter


def _augment(load_path: str, save_path: str, *augmenters_desc: str):
    """Add counterfactuals to the dataset and save it elsewhere.

    Args
    - load-path: the path of the dataset to augment
    - save-path: the path where the augmenter dataset will be save
    - augmenters: a list of ways of converting a string to another string.
                  * If it ends with a .json, assumes it's a the path to a file containing
                  instructions to build a converter. See the docs [LINK] for more info.
                  * Otherwise, assume it is one of the default augmenters: either 'gender' or 'west_v_asia' or 'paraphrase'
                  * If no converter is provided, default to 'gender'

    Example use:
    - countergen augment LOAD_PATH SAVE_PATH gender west_v_asia
    - countergen augment LOAD_PATH SAVE_PATH CONVERTER_PATH
    - countergen augment LOAD_PATH SAVE_PATH gender CONVERTER_PATH
    - countergen augment LOAD_PATH SAVE_PATH
    """

    if not augmenters_desc:
        augmenters_desc = ("gender",)

    augmenters: List[Augmenter] = []
    for c_str in augmenters_desc:
        if c_str.endswith(".json"):
            augmenter = SimpleAugmenter.from_json(c_str)
        elif c_str in DEFAULT_CONVERTERS_PATHS:
            augmenter = SimpleAugmenter.from_default(c_str)
        elif c_str == "paraphrase":
            augmenter = LlmParaphraser()
        else:
            print(f"{c_str} is not a valid augmenter name.")
            return
        augmenters.append(augmenter)
    ds = Dataset.from_jsonl(load_path)
    aug_ds = ds.augment(augmenters)
    aug_ds.save_to_jsonl(save_path)
    print("Done!")


def _evaluate(
    load_path: str,
    save_path: Optional[str] = None,
    model_name: Optional[str] = None,
):
    """Evaluate the provided model.

    Args
    - load-path: the path to the augmented dataset
    - save-path: Optional flag. If present, save the results to the provided path. Otherwise, print the results
    - model-name: Optional flag. Use the model from the openai api given after the flag, or ada is none is provided

    Note: the augmented dataset should match the kind of network you evaluate! See the docs [LINK] for more info.

    Example use:
    - countergen evaluate LOAD_PATH SAVE_PATH
      (use ada and save the results)
    - countergen evaluate LOAD_PATH  --model-name text-davinci-001
      (use GPT-3 and print the results)
    """

    ds = AugmentedDataset.from_jsonl(load_path)
    model_api = api_to_generative_model() if model_name is None else api_to_generative_model(model_name)
    model_ev = get_generative_model_evaluator(model_api)

    evaluate_and_print(ds.samples, model_ev)

    if save_path is not None:
        print("Done!")


def _overwrite_fire_help_text():  # type: ignore
    import inspect
    import fire  # type: ignore
    from fire import inspectutils  # type: ignore
    from fire.core import _ParseKeywordArgs  # type: ignore

    # Replace the default help text by the __doc__
    def NewHelpText(component, trace=None, verbose=False):
        if callable(component):
            return component.__doc__
        elif isinstance(component, dict):
            docs = {k: v.__doc__.split("\n")[0] for k, v in component.items()}
            return "COMMANDS\n" + "\n".join(f"  {k}\n   {doc}" for k, doc in docs.items())
        else:
            return ""

    fire.helptext.HelpText = NewHelpText

    # Remove the INFO line
    def _NewIsHelpShortcut(component_trace, remaining_args):
        show_help = False
        if remaining_args:
            target = remaining_args[0]
            if target in ("-h", "--help"):
                # Check if --help would be consumed as a keyword argument, or is a member.
                component = component_trace.GetResult()
                if inspect.isclass(component) or inspect.isroutine(component):
                    fn_spec = inspectutils.GetFullArgSpec(component)
                    _, remaining_kwargs, _ = _ParseKeywordArgs(remaining_args, fn_spec)
                    show_help = target in remaining_kwargs
                else:
                    members = dict(inspect.getmembers(component))
                    show_help = target not in members
        if show_help:
            component_trace.show_help = True
            # [Where the INFO line was printed]
        return show_help

    fire.core._IsHelpShortcut = _NewIsHelpShortcut


def run():
    _overwrite_fire_help_text()
    fire.Fire(
        {
            "augment": _augment,
            "evaluate": _evaluate,
        },
    )
