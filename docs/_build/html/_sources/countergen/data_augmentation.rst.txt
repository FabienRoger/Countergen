.. _Data Augmentation:

Data Loading and Augmentation
=============================

To produce the :py:class:`AugmentedSample` on which your model will be evaluated, you have two options.

A. Loading default augmented datasets
---------------------------------------

Create an instance of :py:class:`AugmentedDataset` by choosing a default dataset (using the ``AugmentedDataset.from_default`` class method).

.. autoclass:: countergen.AugmentedDataset
   :members: samples, from_default

You can use one of the available default datasets. Possible names are the keys of :py:class:`DEFAULT_AUGMENTED_DS_PATHS`.

Some defaults that you might find interesting:

* "doublebind-heilman-paraphrased" is composed of the double bind experiment questions (by Heilman), augmented through paraphrasing using Instruct GPT (and manually reviewed). Variations are done along the original axis of variation: male vs female.
* "hate-aug-muslim" is composed of hatefull and not hatefull comments. Instruct GPT was used to write variations of this comment where the comment was rewritten to be about Muslims or not about Muslims (this is a variation of LLMD).

B. Loading your data and augmenting it
-----------------------------------------

Data Loading
~~~~~~~~~~~~~

First, you need to load samples.

.. autoclass:: countergen.Sample

This can be done by using on of the utils of the :py:class:`Dataset` class.

.. autoclass:: countergen.Dataset
   :undoc-members:

Data Augmentation : The Augmenter class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Then, you will create variations of your data belonging to different categories. The object which allows you to do that is an :py:class:`Augmenter`: it can transform inputs to any of its target categories.

.. autoclass:: countergen.types.Augmenter
   :undoc-members:

For instance, for an :py:class:`Augmenter`: that can convert to categories "male" and "female", applying the ``transform`` function to "She left the store" with a target category of "male" should output "He left the store", and if the target category is "female", it should return the sentence unchanged.

You can use your :py:class:`Augmenter`: by calling the :py:meth:`Dataset.augment`: method, or by calling directly :py:func:`generate_all_variations`.

.. autofunction:: countergen.generate_all_variations

The two default Augmenters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you a fast and cheap method, use :py:class:`SimpleAugmenter`:, which does word level substitution between two categories by following rules defined in a json file.

.. autofunction:: countergen.SimpleAugmenter

If you a more flexible and powerful method, use :py:class:`LlmdAugmenter`:, which uses a language model to generating variations. This technique has already been used in the context of Counterfactual dataset generation and is called LLMD. You will need to set :py:data:`countergen.config.OPENAI_API_KEY`: to your API key to use it (or set the ``OPENAI_API_KEY`` environment variable).

.. autofunction:: countergen.LlmdAugmenter

Paraphrasing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your dataset is too small, you can make it grow by using a paraphraser, which is just a special type of Augmenter which doesn't have any output category and is special-cased in :py:func:`generate_all_variations`.

One way to do paraphrasing is to use a langue model, and this is already implemented here:

.. autofunction:: countergen.LlmParaphraser