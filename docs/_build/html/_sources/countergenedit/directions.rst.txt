.. _Finding the Relevant Directions:

Finding the Relevant Directions
================================

Using an :py:class:`ActivationsDataset` generated the previous section, you can find along which vectors the difference between variations from one category or the other are most meaningful.

The three methods we provide to find these directions take :py:class:`ActivationsDataset` as inputs, but any PyTorch dataset which ``x_data`` and ``y_data`` of the appropriate types and shapes (see :ref:`ActivationsDataset<ActivationsDataset>`) will work.

.. autofunction:: countergenedit.inlp

.. autofunction:: countergenedit.rlace

.. autofunction:: countergenedit.bottlenecked_mlp_span

:py:func:`bottlenecked_mlp_span` is the fastest method, and :py:func:`rlace` is slow, :py:func:`rlace` is usually better at finding the few directions which matter the most.

:py:func:`inlp` is somewhat in between: it can be very fast if you only want to remove a small number of dimensions, but in contrast to the other two methods, its cost grows linearly with the number of dimensions removed.

Because :py:func:`inlp` is iterative, you can just take the first k directions the function gives you and it will work, whereas this is not true for the other two methods.

There is not support yet for automatically deciding which layers of the network are responsible for those differences.