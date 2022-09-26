Data Loading and Augmentation
=============================

To produce the :py:class:`AugmentedSample` on which your model will be evaluated, you have three options.

A. Loading default augmented datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create an instance of :py:class:`AugmentedDataset` using this class method:

.. automethod:: countergen.AugmentedDataset.from_default

B. Loading your datasets and using
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~