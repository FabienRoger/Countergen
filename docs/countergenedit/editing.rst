Editing the Model
==================

Once you have found the directions which matter the most, you can remove the information stored there by projecting activation along these directions during the forward pass.

This modification in the network are described by a configuration object:

.. autoclass:: countergenedit.ReplacementConfig
   :members:
   :member-order: bysource

Note: ``has_leftover`` is ``False`` in most networks, but if you are using a Transformer from the Huggingface Transformer module, and using :py:func:`get_res_modules`, you should set it to true, as the GPTBlocks have outputs of the form ``(y, attention_mask)``

To generate it in the simple context where you apply the same projection to every module, generate this config using the following:

.. autofunction:: countergenedit.get_edit_configs

Finally, the edit will be done by replacing the target modules in the network by modules which perform the operation of the original modules and the project along the dimensions given by the config. This is done in one simple function call:

.. autofunction:: countergenedit.edit_model

If you don't want to copy the model, for example because you want to spare memory, you can use the inplace version:

.. autofunction:: countergenedit.edit_model_inplace

You can then recover the original model by calling the return the returned handle, or by calling the following function:

.. autofunction:: countergenedit.recover_model_inplace