.. _Model Loading:

Model Loading
=============

What is a ModelEvaluator
-------------------------

The ``countergen`` module is agnostic towards what the model is or does. All it needs is a :py:data:`ModelEvaluator`

.. data:: countergen.types.ModelEvaluator

    Callable that returns the performance of a model given an input and expected outputs.

The performance is usually a float between zero and one, where one is better, but you can use whichever output you want as long as it is supported by the :py:data:`Aggregator` you are using.

It usually is created by plugging a model, which outputs a prediction, into an evaluator, which measures how well the prediction matches the expected outputs.

Create ModelEvaluator from an API
-------------------------------------

To create a :py:data:`ModelEvaluator` using the OpenAI API (or any API compatible with the ``openai`` module), first declare:

* :py:data:`countergen.config.apiconfig.key`: to your API key (or set the ``OPENAI_API_KEY`` environment variable).
* :py:data:`countergen.config.apiconfig.base_url`: to your the URL of the API you want to use (or set the ``OPENAI_API_BASE_URL`` environment variable). Defaults to the OpenAI API URL.

Then create a generative model using the following function:

.. autofunction:: countergen.api_to_generative_model

Finally, use this generative model to create the model evaluator:

.. autofunction:: countergen.get_generative_model_evaluator

Note: instead of declaring global API configurations, you can also pass a :py:class:`ApiConfig`: object:

.. autoclass:: countergen.tools.api_utils.ApiConfig

Create ModelEvaluator from a local model
------------------------------------------

See ``countergenedit`` which contains utilities to build :py:data:`ModelEvaluator` from PyTorch generative and classification models.

Examples of ModelEvaluator
--------------------------

If you are using Tensorflow or Keras, or if your model takes intput different from thoses of Huggingface transformers take, just create the :py:data:`ModelEvaluator` directly.

If you are evaluating classification models, :py:data:`ModelEvaluator` you might use in practice is a function which compute the probability of the correct label given the input:

>>> def typical_classification_model_evaluator(input: str, outputs: List[str]) -> float:
>>>     """ModelEvaluator for a generative model"""
>>>     correct_label = outputs[0] # Excepts excalty only one output
>>>     input_tokens = tokenize(input)
>>>     labels_probabilities = model(input_tokens) # Compute the predictions of the model
>>>     return labels_probabilities[correct_label] # Return the probability of the correct label

If you are evaluating generative models, :py:data:`ModelEvaluator`, you might use a function which compute the probability of each output given the input, and return the sum of those:

>>> def typical_generative_model_evaluator(input: str, outputs: List[str]) -> float:
>>>     """ModelEvaluator for a generative model."""
>>>     input_tokens = tokenize(input)
>>>     outputs_probabilities = []
>>>     for output in outputs:
>>>         output_tokens = tokenize(output)
>>>         # Logits of each token in the input and output
>>>         logits = model(input_tokens + output_tokens) 
>>>         # Logits of each token at each position of the output
>>>         output_logits = logits[len(input_tokens):] 
>>>         # Probability of each token at each position of the output
>>>         probabilities = softmax(output_logits) 
>>>         # Probability of the correct token at each position of the output
>>>         output_probabilities = [
>>>             probs[tok] for tok, probs in zip(output_tokens, output_probabilities)
>>>         ]
>>>         # Probability of the whole output
>>>         output_probability = product(output_correct_probabilities)
>>>         outputs_probabilities.append(output_probability)
>>>     return sum(outputs_probabilities)

You can also adapt the code above to the case where you call an API (different from the openai API that countergen natively supports). If you need help, checkout how :py:func:countergen.api_to_generative_model: is implemented.