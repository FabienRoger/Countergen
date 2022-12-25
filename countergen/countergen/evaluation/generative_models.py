from math import exp
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple

import countergen.config
import openai
from countergen.tools.api_utils import ApiConfig
from countergen.types import Input, ModelEvaluator, Outputs, Performance

metrics = ["perplexity", "probability"]

LogProbs = float

# Return a log prob for each token of output
GenerativeModel = Callable[[Input, Outputs], Sequence[Sequence[LogProbs]]]


def api_to_generative_model(
    openai_engine: str = "ada", apiconfig: Optional[ApiConfig] = None, max_attempts: Optional[int] = 5
) -> GenerativeModel:
    """Make a GenerativeModel that uses the openai api.

    The resulting GenerativeModel takes as input an input text and possibles outputes,
    and returns the log probabilities of each tokens of each expected output.

    The GenerativeModel costs ~ len(input) * (sum of len(ouput)) tokens per call.

    If the api call fails, it will retry max_attempts times, or forever if max_attempts is None."""

    def gen_model(inp: Input, out: Outputs) -> List[List[float]]:
        apiconfig_ = apiconfig or countergen.config.apiconfig

        correct_log_probs_list = []
        for o in out:

            completion: Optional[dict] = None
            attemps = 0
            while completion is None:
                try:
                    completion = openai.Completion.create(
                        engine=openai_engine,
                        prompt=inp + o,
                        max_tokens=1,
                        stream=False,
                        echo=True,
                        logprobs=5,
                        **apiconfig_.get_config(),
                    )["choices"][0]
                except Exception as e:
                    attemps += 1
                    if max_attempts is not None and attemps >= max_attempts:
                        raise e

            token_logprobs = completion["logprobs"]["token_logprobs"]
            token_offsets = completion["logprobs"]["text_offset"]

            # Compute which tokens correspond to the output
            # If token from input & output got merged (which should happen very rarely),
            # takes into account the proability of the merged token.
            n_toks = len(token_offsets)
            start_of_output = max([i for i in range(n_toks) if token_offsets[i] <= len(inp)])

            correct_log_probs_list.append(token_logprobs[start_of_output:-1])
        return correct_log_probs_list

    return gen_model


def get_generative_model_evaluator(model: GenerativeModel, metric: str = "probability") -> ModelEvaluator:
    """Return the ModelEvaluator corresponding to the model & the metric.

    Available metrics: probability & perplexity"""

    def run(inp: Input, out: Outputs) -> Performance:
        if len(out) == 0:
            raise ValueError("Expected output should be provided for gpt models")
        if len(inp) == 0:
            raise ValueError("Empty inputs are forbidden for gpt models.")

        correct_log_probs_list = model(inp, out)

        total_prob: float = 0
        total_log_prob: float = 0
        number_of_toks: int = 0
        for correct_log_probs in correct_log_probs_list:
            total_prob += exp(sum(correct_log_probs))
            number_of_toks += len(correct_log_probs)
            total_log_prob += sum(correct_log_probs)

        if metric == "perplexity":
            return exp(-total_log_prob / number_of_toks)
        if metric == "probability":
            return total_prob
        raise ValueError(f"{metric} is not a valid metric. Choose one in {metrics}.")

    return run
