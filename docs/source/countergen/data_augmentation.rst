Data Loading and Augmentation
=============================

To produce the :py:class:`AugmentedSample` on which your model will be evaluated, you have three options.

A. Loading default augmented datasets
---------------------------------------

Create an instance of :py:class:`AugmentedDataset` using this class method:

.. automethod:: countergen.AugmentedDataset.from_default

The available datasets are the keys of :py:class:`DEFAULT_AUGMENTED_DS_PATHS`.

Some defaults that you might find interesting:

* "doublebind-heilman-paraphrased" is composed of the double bind experiment questions (by Heilman), augmented through paraphrasing using Instruct GPT (and manually reviewed). Variations are done along the original axis of variation: male vs female.
* "hate-aug-muslim" is composed of hatefull and not hatefull comments. Instruct GPT was used to write variations of this comment where the comment was rewritten to be about Muslims or not about Muslims (this is a variation of LLMD).

B. Loading your datasets and using your data
---------------------------------------------