Welcome to Countergen's documentation!
======================================

**Countergen** is a framework for generating counterfactual datasets,
evaluating NLP models on the variations and editing models to reduce unwanted bias.
It offers simple ways to generate counterfactual datasets using state-of-the-art techniques,
while offering simple ways to add your own data, data augmentation techniques, models, and evaluation metrics.

This project is split in two parts:

* `countergen` is a lightweight Python module which provides the framework for data augmentation and model evaluation. It provides useful default datasets, lightweight augmentation techniques, and evaluation of models available through an API.
* `countergentorch ` is a Python module which adds methods to easily evaluate PyTorch text generation models as well as text classifiers. It provides tools to analyze model activation and quickly edit model to reduce bias.

Check out the :doc:`/countergen/usage` section for further information, including
how to :ref:`installation` the project.

.. note::

   This project is under active development.

Contents
--------

.. toctree::

   /countergen/usage

.. toctree::
   :caption: countergen

   /countergen/usage

.. toctree::
   :caption: countergentorch

   /countergen/usage