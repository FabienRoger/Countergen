from typing import Dict, Optional
from attrs import define
from countergen.config import DEFAULT_API_BASE_URL
from countergen.tools.utils import unwrap_or


@define
class ApiConfig:
    """Hold API key and API URL"""

    key: Optional[str] = None
    base_url: str = DEFAULT_API_BASE_URL

    def get_config(self) -> Dict[str, str]:
        """Return the argument the openai module needs."""

        if self.key is None:
            raise RuntimeError(
                "Please provide openai key to use its api! Use `countergen.config.apiconfig.key = YOUR_KEY`"
            )
        return {
            "api_key": self.key,
            "api_base": self.base_url,
        }
