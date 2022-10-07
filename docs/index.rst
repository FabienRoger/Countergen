Welcome to Countergen's documentation!
======================================

**Countergen** is a framework for generating counterfactual datasets,
evaluating NLP models on the variations and editing models to reduce unwanted bias.
It offers simple ways to generate counterfactual datasets using state-of-the-art techniques,
while offering simple ways to add your own data, data augmentation techniques, models, and evaluation metrics.

The project is free and open source. The code can be found here: https://github.com/FabienRoger/Countergen.

This project is split in two parts:

* ``countergen`` is a lightweight Python module which provides the framework for data augmentation and model evaluation. It provides useful default datasets, lightweight augmentation techniques, and evaluation of models available through an API. It enable 
* ``countergentorch`` is a Python module which adds methods to easily evaluate PyTorch text generation models as well as text classifiers. It provides tools to analyze model activation and quickly edit model to reduce bias.

Check out the :doc:`/countergen/usage` section for further information, including
how to :ref:`installation` the project.


Workflow
--------

Here is the global workflow to evaluate model bias use ``countergen``:

.. image:: evaluation_workflow.png
  :width: 600
  :align: center
  :alt: Evaluation worflow

|

And here is how you can use ``countergentorch`` to edit models:

.. image:: edition_workflow.png
  :width: 600
  :align: center
  :alt: Edition worflow

|

.. note::

   This project is under development. Please report issues in the appropriate section of the Github repository.

.. note::
   This module only works for Python version 3.7 and above

Contents
--------

.. toctree::
   demos

.. toctree::
   :maxdepth: 2
   :caption: countergen

   /countergen/usage
   /countergen/data_augmentation
   /countergen/model_loading
   /countergen/model_evaluation

.. toctree::
   :maxdepth: 1
   :caption: countergentorch

   /countergentorch/usage
   /countergentorch/model_loading
   /countergentorch/internal_bias
   /countergentorch/directions
   /countergentorch/editing