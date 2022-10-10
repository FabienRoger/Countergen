Model Evaluation
================

Aggregator Selection
---------------------------------

To make sense of the performances of the model on every sample, you need to aggregate them.

This operation is performed by an object that inherit from the following abstract class.

.. autoclass:: countergen.types.Aggregator

``countergen`` provides a few useful defaults, and the most useful ones are probably the following:

.. autoclass:: countergen.aggregators.PerformanceStatsPerCategory

.. autoclass:: countergen.aggregators.DifferenceStats

.. autoclass:: countergen.aggregators.OutliersAggregator

.. autoclass:: countergen.aggregators.BiasFromProbsAggregator

.. autoclass:: countergen.aggregators.AveragePerformancePerCategory

If you want to create your own aggregator, here is the type of data you will aggregate over:

>>> # Performance is either the performance over outputs, or the performance on every output
>>> # (An aggregator can handle only one of these and raise a ValueError in the other case)
>>> # usually between zero & one (one is better)
>>> Performance = Union[float, List[float]]
>>> VariationResult = Tuple[Performance, Tuple[Category, ...]]
>>> SampleResults = Sequence[VariationResult]
>>> Results = Sequence[SampleResults] # Input to the aggregator ``__call__`` function

Model Evaluation
---------------------------------

Once you have your model evaluator, your augmented samples, and you have chosen how to aggregate the performances, simply call the evalute function!

.. autofunction:: countergen.evaluate

You can print or save the results using the :py:meth:`Aggregator.save_aggregation` or by calling on of the following:

.. autofunction:: countergen.evaluate_and_print

.. autofunction:: countergen.evaluate_and_save

If you want to use multiple aggregators, first compute the results using :py:func:`compute_performances`, and then call each aggregator on the result

.. autofunction:: countergen.compute_performances