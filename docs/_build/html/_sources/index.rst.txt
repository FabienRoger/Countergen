Welcome to CounterGen's documentation!
======================================

**CounterGen** is a framework for generating counterfactual datasets, evaluating NLP models, and editing models to reduce bias.
It provides powerful defaults, while offering simple ways to use your own data, data augmentation techniques, models, and evaluation metrics.

The project is free and open source. The code can be found here: https://github.com/FabienRoger/Countergen.

This project is split in two parts:

* ``countergen`` is a lightweight Python module which helps you generate counterfactual datasets and evaluate model bias. It provides useful default datasets, lightweight augmentation techniques, and evaluation of models available through an API.
* ``countergenedit`` is a Python module which adds methods to easily evaluate PyTorch text generation models as well as text classifiers. It provides tools to analyze model activation and quickly edit model to reduce bias.

Check out the :doc:`countergen usage </countergen/usage>` and the :doc:`countergenedit usage </countergenedit/usage>` for more information about installation and usage of these modules.



.. note::

   This project is under development. Please report issues in the appropriate section of the Github repository.

.. note::
   This module only works for Python version 3.7 and above

Contents
--------

.. toctree::
   demos
   citations

.. toctree::
   :maxdepth: 2
   :caption: countergen

   /countergen/usage
   /countergen/data_augmentation
   /countergen/model_loading
   /countergen/model_evaluation

.. toctree::
   :maxdepth: 1
   :caption: countergenedit

   /countergenedit/usage
   /countergenedit/model_loading
   /countergenedit/internal_bias
   /countergenedit/directions
   /countergenedit/editing