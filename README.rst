CounterGen
==========

**CounterGen** is a framework for generating counterfactual datasets, evaluating NLP models, and editing models to reduce bias.
It provides powerful defaults, while offering simple ways to use your own data, data augmentation techniques, models, and evaluation metrics.

* ``countergen`` is a lightweight Python module which helps you generate counterfactual datasets and evaluate bias of models available locally or through an API. The generated data can be used to finetune the model on mostly debiased data, or can be injected into ``countergentorch`` to edit the model directly.
* ``countergentorch`` is a Python module which adds methods to easily evaluate PyTorch text generation models as well as text classifiers. It provides tools to analyze model activation and edit model to reduce bias.

Read the docs here: https://fabienroger.github.io/Countergen/

Use the tool online here: https://www.safer-ai.org/countergenweb

Minimal examples
---------------------

Model evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

>>> import countergen
>>> augmented_ds = countergen.AugmentedDataset.from_default("male-stereotypes")
>>> api_model = countergen.api_to_generative_model("text-davinci-001")
>>> model_evaluator = countergen.get_generative_model_evaluator(api_model)
>>> countergen.evaluate_and_print(augmented_ds.samples, model_evaluator)

*(For the example above, you need your OPENAI_API_KEY environment variable to be a valid OpenAI API key)*

Data augmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

>>> import countergen
>>> ds = countergen.Dataset.from_jsonl("my_data.jsonl")
>>> augmenters = [countergen.SimpleAugmenter.from_default("gender")]
>>> augmented_ds = ds.augment(augmenters)
>>> augmented_ds.save_to_jsonl("my_data_augmented.jsonl")

Model editing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

>>> import countergen as cg
>>> import countergentorch as cgt
>>> from transformers import GPT2LMHeadModel
>>> augmented_ds = cg.AugmentedDataset.from_default("male-stereotypes")
>>> model = GPT2LMHeadModel.from_pretrained("gpt2")
>>> layers = cgt.get_mlp_modules(model, [2, 3])
>>> activation_ds = cgt.ActivationsDataset.from_augmented_samples(
>>>   augmented_ds.samples, model, layers
>>> )
>>> # INLP is an algorithm to find important directions in a dataset
>>> dirs = cgt.inlp(activation_ds)
>>> configs = cgt.get_edit_configs(layers, dirs, has_leftover=False)
>>> new_model = cgt.edit_model(model, configs=configs)

How ``countergen`` helps you evaluate model bias
----------------------------------------------

.. image:: docs/evaluation_workflow.png
  :width: 600
  :align: center
  :alt: Evaluation worflow

|

How ``countergentorch`` helps you edit models
----------------------------------------------

.. image:: docs/edition_workflow.png
  :width: 600
  :align: center
  :alt: Edition worflow

|