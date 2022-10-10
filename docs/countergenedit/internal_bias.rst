Measuring Internal Bias
========================

The ActivationsDataset
------------------------

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

Which modules should I select? How?
--------------------------------------

In Transformers, measuring the output of layers in the middle of the network, and doing the editing (i.e. the projection) in the residual stream usually works best. If you are using a Huggingface GPT model, you can use this function to select these layers:

.. autofunction:: countergenedit.get_res_modules

You can also select the output of MLPs, which will not affect the residual stream directly. If you do so, it is advised to edit multiple layers at once. If you are using a Huggingface GPT model, you can use this function to select MLP layers:

.. autofunction:: countergenedit.get_mlp_modules