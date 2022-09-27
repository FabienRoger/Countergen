Model loading
=============

What is a ModelEvaluator
-------------------------

The ``countergen`` module is agnostic towards what the model is or does. All it needs is a :py:data:`ModelEvaluator`

.. data:: countergen.types.ModelEvaluator

    Callable that returns the performance of a model given an input and expected outputs.

The performance is usually a float between zero and one, where one is better, but you can use whichever output you want as long as it is supported by the :py:data:`StatsAggregator` you are using.

It usually is created by plugging a model, which outputs a prediction, into an evaluator, which measure how well the prediction matches the expected outputs.

Create ModelEvaluator from an API
-------------------------------------

To create a :py:data:`ModelEvaluator` using the OpenAI API (or any API compatible with the ``openai`` module), first declare:

* :py:data:`countergen.config.OPENAI_API_KEY`: to your API key (or set the ``OPENAI_API_KEY`` environment variable).
* :py:data:`countergen.config.OPENAI_API_BASE`: to your the URL of the API you want to use (or set the ``OPENAI_API_BASE`` environment variable). Defaults to the OpenAI API URL.

Then create a generative model using the following function:

.. autofunction:: countergen.api_to_generative_model

Finally, use this generative model to create the model evaluator:

.. autofunction:: countergen.get_generative_model_evaluator

Create ModelEvaluator from a local model
------------------------------------------

See ``countergentorch`` which contains utilities to build :py:data:`ModelEvaluator` from PyTorch generative and classification models.