import os
from pathlib import Path
from countergen.tools.api_utils import ApiConfig

from dotenv import load_dotenv

load_dotenv()

verbose = int(os.environ.get("CDG_VERBOSE", "0"))

MODULE_PATH = str(Path(__file__).parent)  # To load internal data

DEFAULT_API_BASE_URL = "https://api.openai.com/v1"

apiconfig = ApiConfig(
    key=os.environ.get("OPENAI_API_KEY", None),
    base_url=os.environ.get("OPENAI_API_BASE_URL", DEFAULT_API_BASE_URL),
)
