Outline of the evaluation process
=================================

Data Augmentation
-----------------

First, you produce a list of :py:class:`AugmentedSample`, either by loading an existing one, or using tools the library provides to build you it from raw data, or by creating your own from scratch.

.. autoclass:: countergen.types.AugmentedSample
   :members:

Where a :py:class:`Variation` is defined as follows.

.. autoclass:: countergen.types.Variation
   :members: text, categories
   :member-order: bysource

Model Loading
-----------------

Second, you load your model and turn it into a :py:data:`ModelEvaluator`.

.. data:: countergen.types.ModelEvaluator

    Callable that returns the performance of a model given an input and expected outputs.

Model Evaluation
-----------------

Third, you pass your list of :py:class:`AugmentedSample` and your :py:data:`ModelEvaluator` into the following function:

.. autofunction:: countergen.evaluate

By default, it will return the average performance on each kind of data. You can specify other ways to aggregate the performance on each variation by passing another :py:class:`StatsAggregator`.

Alternatively, if you just want to print or save the results, directly call on of the following:

.. autofunction:: countergen.evaluate_and_print

.. autofunction:: countergen.evaluate_and_save

And that's it! ``countergen`` and ``countergentorch`` provides to generate variations, load models easily, and edit them to decrease performance gaps between different categories. Click on *Next* to know more.