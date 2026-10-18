"""Shared configuration. Environment variables take precedence over .env."""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[1] / '.env')
BASE_URL = os.getenv('BASE_URL', '')
MODEL = os.getenv('MODEL', '')
API_KEY = os.getenv('API_KEY', '')
# The original scripts use server defaults: no sampling overrides.
SAMPLING_PARAMETERS = {}


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f'Missing {name}. Copy .env.example to .env and set it.')
    return value


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    for name in ('BASE_URL', 'MODEL', 'API_KEY'):
        require_env(name)
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


def check_server() -> tuple[bool, str]:
    try:
        get_client().with_options(timeout=3, max_retries=0).models.list()
        return True, 'ok'
    except Exception as exc:
        # Do not display request URLs, credentials or server response bodies.
        reason = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
        return False, f'Server unavailable ({reason}). Check BASE_URL and your server.'
