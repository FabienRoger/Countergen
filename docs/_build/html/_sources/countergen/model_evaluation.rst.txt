Model evaluation
================

Aggregator Selection
---------------------------------

To make sense of the performances of the model on every sample, you need to aggregate them.

This operation is performed by an object that inherit from the following abstract class.

.. autoclass:: countergen.types.StatsAggregator

``countergen`` provides a few useful defaults, and the most useful ones are probably the following:

.. autoclass:: countergen.aggregators.AveragePerformancePerCategory

.. autoclass:: countergen.aggregators.PerformanceStatsPerCategory

Model Evaluation
---------------------------------

Once you have your model evaluator, your augmented samples, and you have chosen how to aggregate the performances, simply call the evalute function!

.. autofunction:: countergen.evaluate

You can print or save the results using the :py:meth:`StatsAggregator.save_aggregation` or by calling on of the following:

.. autofunction:: countergen.evaluate_and_print

.. autofunction:: countergen.evaluate_and_save