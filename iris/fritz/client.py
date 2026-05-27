"""
iris.fritz.client
=================
Low-level HTTP client for the Fritz / SkyPortal API.
"""

import time
import urllib.parse
import logging
import requests

logger = logging.getLogger(__name__)


class FritzClient:

    def __init__(
        self,
        token: str,
        base_url: str = "https://fritz.science",
        max_retries: int = 6,
        backoff_base: float = 2.0,
    ):
        self.base_url     = base_url.rstrip("/") + "/"
        self.max_retries  = max_retries
        self.backoff_base = backoff_base
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"token {token}"})

    def request(self, method: str, endpoint: str, **kwargs):
        endpoint = endpoint.lstrip("/")
        url = urllib.parse.urljoin(self.base_url, endpoint)

        for attempt in range(self.max_retries + 1):
            r = self.session.request(method.upper(), url, timeout=60, **kwargs)

            if r.status_code == 429:
                wait = self.backoff_base ** attempt
                logger.warning(
                    f"429 on {endpoint} — waiting {wait:.1f}s "
                    f"(attempt {attempt+1}/{self.max_retries})"
                )
                time.sleep(wait)
                continue

            r.raise_for_status()
            return r.json().get("data", {})

        raise RuntimeError(f"Exceeded max retries for {endpoint}")

    def get_source(self, obj_id: str) -> dict:
        return self.request("GET", f"api/sources/{obj_id}")

    def list_sources(self, page: int = 1, num_per_page: int = 100, **filters) -> dict:
        params = {"pageNumber": page, "numPerPage": num_per_page, **filters}
        return self.request("GET", "api/sources", params=params)

    def get_photometry(self, obj_id: str) -> list:
        return self.request("GET", f"api/sources/{obj_id}/photometry")

    def get_spectra(self, obj_id: str) -> list:
        data = self.request("GET", f"api/sources/{obj_id}/spectra")
        return data.get("spectra", [])
