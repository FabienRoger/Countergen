PyTorch Model Loading
======================

Just as :py:func:`countergen.api_to_generative_model` and :py:func:`countergen.get_generative_model_evaluator` allow you to create a :py:data:`ModelEvaluator` (see :ref:`here <Model Loading>` if you haven't read about it yet), ``countergenedit``  provides two similar functions which work for PyTorch models, while providing optimization which only work on model you run locally (for 1-token outputs, having access to the full logit vector allows you to run the model only once no matter how many outputs you expect).

.. autofunction:: countergenedit.pt_to_generative_model

.. autofunction:: countergenedit.get_generative_model_evaluator

You can also evaluate classification models by using the following function:

.. autofunction:: countergenedit.get_classification_model_evaluator

Finally, if you just want to evaluate a pipeline from Hugginface's ``transformers`` library, you can use

.. autofunction:: countergenedit.get_classification_pipline_evaluator

`Note: this last function won't let you access the weights of your model, making editing fail with the tools provided by this library.`