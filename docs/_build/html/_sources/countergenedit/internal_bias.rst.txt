Measuring Internal Bias
========================

To measure how the internal activations of your model vary across categories, you first need to generate your :py:class:`AugmentedSample` (see :ref:`Data Augmentation<Data Augmentation>`).

Then, you quickly measure the activations and turn them into a PyTorch dataset using the following class:

.. _ActivationsDataset:

.. autoclass:: countergenedit.ActivationsDataset
   :members: x_data, y_data, from_augmented_samples
   :noindex:
   :undoc-members:

This will group activations from different parts of the network together.

Multiclass classification dataset are not yet supported.

If you want something more precise, you can build the dataset yourself by using the following function

.. autofunction:: countergenedit.editing.activation_utils.get_activations