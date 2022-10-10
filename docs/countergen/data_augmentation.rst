.. _Data Augmentation:

Data Loading and Augmentation
=============================

To produce the :py:class:`AugmentedSample` on which your model will be evaluated, you have two options.

A. Loading default augmented datasets
---------------------------------------

Create an instance of :py:class:`AugmentedDataset` by choosing a default dataset (using the ``AugmentedDataset.from_default`` class method).

.. autoclass:: countergen.AugmentedDataset
   :members: samples, from_default
   :noindex:

You can use one of the available default datasets. Possible names are the keys of :py:class:`DEFAULT_AUGMENTED_DS_PATHS`.

Some defaults that you might find interesting:

* ``female-stereotypes`` and ``male-stereotypes``, which are stereotypes from the `StereoSet <https://stereoset.mit.edu/>`_, filtered for the ones whose outputs are gender neutral.
* ``doublebind-likable-paraphrased`` is composed of data from the "Double bind experiment" `(Heilman, 2007) <https://www.researchgate.net/publication/6575591_Why_Are_Women_Penalized_for_Success_at_Male_Tasks_The_Implied_Communality_Deficit>`_ which found that female managers were perceived as less likable than their male counterparts. It is augmented through paraphrasing using Instruct GPT (and manually reviewed), with positive prosocial adjectives. Variations are done along the original axis of variation: male vs female. Exact data is from `May (2019) <https://arxiv.org/abs/1903.10561>`_, which measures this kind of bias in Language models.
* ``doublebind-unlikable-paraphrased`` the same as above, but with negative antisocial adjectives.
* ``bias-from-probs`` is the dataset of variations by BigBench's `Social Bias from Sentence Probability <https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/bias_from_probabilities>`_. The output associated with each output aren't in a particular direction (bias towards or against minorities), but is rather made to emphasize that sometimes, the output of the model will be very different from one variation to the next. It doesn't have clear cut categories and is made for evaluation only.
* ``simple-gender`` is a simple dataset with a of 1-token variations per sample, making it ideal to find gender-related directions in activation space.
* ``hate-aug-muslim`` is composed of comments labeled "hate" or "noHate". Instruct GPT was used to write variations of this comment where the comment was rewritten to be about Muslims or not about Muslims (this is a variation of LLMD). Data is from `De Gibert (2018) <https://arxiv.org/abs/1809.04444>`_.

You can also use one of the defaults datasets and augment them.

B. Loading your data and augmenting it
-----------------------------------------

Data Loading
~~~~~~~~~~~~~

First, you need to load samples.

.. autoclass:: countergen.types.Sample

This can be done by using on of the utils of the :py:class:`Dataset` class.

.. autoclass:: countergen.Dataset
   :members:

Data Augmentation : The Augmenter class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Then, you will create variations of your data belonging to different categories. The object which allows you to do that is an :py:class:`Augmenter`: it can transform inputs to any of its target categories.

.. autoclass:: countergen.types.Augmenter
   :members:

For instance, for an :py:class:`Augmenter`: that can convert to categories "male" and "female", applying the ``transform`` function to "She left the store" with a target category of "male" should output "He left the store", and if the target category is "female", it should return the sentence unchanged.

You can use your :py:class:`Augmenter`: by calling the :py:meth:`Dataset.augment`: method, which will give your an :py:class:`AugmentedDataset`:

.. autoclass:: countergen.types.AugmentedDataset
   :members:

You can also call :py:func:`generate_all_variations` directly.

.. autofunction:: countergen.generate_all_variations

The two default Augmenters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you a fast and cheap method, use :py:class:`SimpleAugmenter`:, which does word level substitution between two categories by following rules defined in a json file.

.. autoclass:: countergen.SimpleAugmenter
   :members:

If you a more flexible and powerful method, use :py:class:`LlmdAugmenter`:, which uses a language model to generating variations. This technique has already been used in the context of Counterfactual dataset generation and is called LLMD. You will need to set :py:data:`countergen.config.apiconfig.key`: to your API key to use it (or set the ``OPENAI_API_KEY`` environment variable).

.. autoclass:: countergen.LlmdAugmenter
   :members:

Paraphrasing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your dataset is too small, you can make it grow by using a paraphraser, which is just a special type of Augmenter which doesn't have any output category and is special-cased in :py:func:`generate_all_variations`.

One way to do paraphrasing is to use a langue model, and this is already implemented here:

.. autoclass:: countergen.LlmParaphraser

Example of Augmenter
~~~~~~~~~~~~~~~~~~~~~~~

If you want to generate variations different from the ones available by defaults, you can easily do so by creating a class inheriting from :py:class:`Augmenter`.

Here is a simple example of an Augmenter which replaces zip codes

>>> class ZipCodeAugmenter(Augmenter):
>>>     def __init__(self) -> None:
>>>         self.categories = [f"zipcode_starts_with:{i}" for i in range(10)]
>>> 
>>>     def transform(self, inp: Input, to: Category) -> Input:
>>>         # Find the zip codes
>>>         zip_code_positions = find_zipcode_positions(inp)
>>>         characters = [c for c in inp]
>>>         # The "to" category is of the form zipcode_starts_with:<NUMBER>
>>>         # Because the target category is always chosen to be a member of self.category
>>>         zipcode_starts_with, zipcode_start = to.split(":")
>>>         # Replace the zip coddes
>>>         for position in zip_code_positions:
>>>             characters[position] = zipcode_start
>>>         return "".join(characters)

*Note: this will create ten variations on each sample. If you want to create variations with a random subset of each zipcode, you can implement a "categories" property method which returns a random subsets of zipcodes.*