from typing import Dict, Optional
from attrs import define
from countergen.tools.utils import FromAndToJson


@define
class ApiConfig(FromAndToJson):
    """Hold API key and API URL"""

    key: Optional[str] = None
    base_url: Optional[str] = None

    def get_config(self) -> Dict[str, str]:
        """Return the argument the openai module needs."""

        if self.key is None:
            raise RuntimeError(
                "Please provide openai key to use its api! Use `countergen.config.apiconfig.key = YOUR_KEY`"
            )
        if self.base_url is None:
            raise RuntimeError(
                "Please provide openai base url to use its api! Use `countergen.config.apiconfig.base_url = BASE_URL`"
            )

        return {
            "api_key": self.key,
            "api_base": self.base_url,
        }
