import requests
from huggingface_hub import configure_http_backend

from HF_pipeline.model import configure_insecure_hub_download


def test_insecure_download_backend_disables_session_verification():
    from huggingface_hub import get_session

    try:
        configure_insecure_hub_download()
        assert get_session().verify is False
    finally:
        configure_http_backend(backend_factory=requests.Session)
