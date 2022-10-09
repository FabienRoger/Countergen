Demos
=====

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
>>> configs = cgt.get_edit_configs(layers, dirs)
>>> new_model = cgt.edit_model(model, configs=configs)

Colab notebook
--------------------

Here is a demo of the library in a Colab notebook:

https://colab.research.google.com/drive/1J6zahRfPfqSyXlA1hm_KQCQlkcd3KVPc