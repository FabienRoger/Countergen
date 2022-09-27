Editing the Model
==================

Once you have found the directions which matter the most, you can remove the information stored there by projecting activation along these directions during the forward pass.

This modification in the network are described by a configuration object:

.. autoclass:: countergentorch.ReplacementConfig
   :members:
   :member-order: bysource

To generate it in the simple context where you apply the same projection to every module, generate this config using the following:

.. autofunction:: countergentorch.get_edit_configs

Finally, the edit will be done by replacing the target modules in the network by modules which perform the operation of the original modules and the project along the dimensions given by the config. This is done in one simple function call:

.. autofunction:: countergentorch.edit_model