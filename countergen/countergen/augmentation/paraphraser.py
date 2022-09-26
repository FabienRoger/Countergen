from typing import Dict, Tuple

import openai
from attrs import define
from countergen.config import OPENAI_API_KEY
from countergen.tools.utils import estimate_paraphrase_length, set_and_check_oai_key
from countergen.types import Augmenter, Category, Input, Paraphraser


# Examples from https://www.pragnakalp.com/intent-classification-paraphrasing-examples-using-gpt-3/
DEFAULT_TEMPLATE = """Paraphrase the original passage.

Original: {Searching a specific search tree for a binary key can be programmed recursively or iteratively.}
Paraphrase: {Searching a specific search tree according to a binary key can be recursively or iteratively programmed.}

Original: {It was first released as a knapweed biocontrol in the 1980s in Oregon , and it is currently established in the Pacific Northwest.}
Paraphrase: {It was first released as Knopweed Biocontrol in Oregon in the 1980s , and is currently established in the Pacific Northwest.}

Original: {4-OHT binds to ER , the ER / tamoxifen complex recruits other proteins known as co-repressors and then binds to DNA to modulate gene expression.}
Paraphrase: {The ER / Tamoxifen complex binds other proteins known as co-repressors and then binds to DNA to modulate gene expression.}

Original: {In mathematical astronomy, his fame is due to the introduction of the astronomical globe, and his early contributions to understanding the movement of the planets.}
Paraphrase: {His fame is due in mathematical astronomy to the introduction of the astronomical globe and to his early contributions to the understanding of the movement of the planets.}

Original: {__input__}
Paraphrase: {"""


@define
class LlmParaphraser(Paraphraser):
    prompt_template: str = DEFAULT_TEMPLATE
    engine: str = "text-davinci-002"

    def transform(self, inp: Input, to: Category = "") -> Input:
        set_and_check_oai_key()

        prompt = self.prompt_template.replace("__input__", inp)

        completion = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            max_tokens=estimate_paraphrase_length(inp),
            temperature=1,
            top_p=0.7,  # LLM-D has top_k=40, but not available
            stream=False,
        )["choices"][0]["text"]

        return completion.split("}")[0]
