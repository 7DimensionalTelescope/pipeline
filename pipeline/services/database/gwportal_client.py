"""
HTTP client for the GWPortal REST API.

This is the fallback transport for the comprehensive DB wrapper. It is adapted
from the official ``gwportal_client.py`` toolkit, with a few local changes:

* Credentials are pulled from ``.const`` (which reads the pipeline ``.env``),
  not directly from the process environment.
* The connection-test request in ``__init__`` is lazy: we do not hit the server
  until the caller actually issues a query. This keeps module import cheap
  and avoids noisy stderr messages during normal imports.
* A ``stream_all_results`` generator is retained for memory-efficient paging.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional
from urllib.parse import urljoin

import requests
from requests.exceptions import JSONDecodeError, RequestException

from .const import GWPORTAL_API_KEY, GWPORTAL_BASE_URL


class GWPortalClient:
    """
    Thin wrapper around the function-based GWPortal REST API.

    Parameters
    ----------
    base_url : str, optional
        Overrides ``GWPORTAL_BASE_URL`` from ``.env``.
    api_key : str, optional
        Overrides ``GWPORTAL_API_KEY`` from ``.env``.
    timeout : float, default 60
        Per-request timeout in seconds.
    """

    DEFAULT_PAGE_SIZE = 500

    # Map a logical entity name to the API path (relative to ``base_url/api/``).
    ENDPOINTS: Dict[str, str] = {
        "raw": "frames/raw/",
        "processed": "frames/processed/",
        "combined": "frames/combined/",
        "processed_too": "frames/processed_too/",
        "combined_too": "frames/combined_too/",
        "tile": "tiles/",
        "target": "targets/",
        "master_bias": "master/bias/",
        "master_dark": "master/dark/",
        "master_flat": "master/flat/",
    }

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url or GWPORTAL_BASE_URL
        self.api_key = api_key or GWPORTAL_API_KEY
        self.timeout = timeout

        if not self.base_url:
            raise ValueError("GWPORTAL_BASE_URL is not configured.")
        if not self.api_key:
            raise ValueError("GWPORTAL_API_KEY is not configured.")

        if not self.base_url.endswith("/"):
            self.base_url += "/"

        self.session = requests.Session()
        self.session.headers.update(
            {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    # ------------------------------------------------------------------ #
    # Low-level request helper
    # ------------------------------------------------------------------ #
    def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        url = urljoin(self.base_url, "api/" + endpoint.lstrip("/"))
        try:
            resp = self.session.get(
                url, params=params or {}, timeout=timeout or self.timeout
            )
            resp.raise_for_status()
            if not resp.content:
                return {}
            return resp.json()
        except JSONDecodeError:
            raise RequestException(
                f"Non-JSON response from {url}. Status={resp.status_code}. "
                f"Body={resp.text[:200]}"
            )
        except RequestException as exc:
            if exc.response is not None:
                try:
                    detail = exc.response.json().get("error", "Unknown API Error")
                except Exception:
                    detail = exc.response.text[:200]
                raise RequestException(
                    f"API Error ({exc.response.status_code}): {detail} [{url}]"
                ) from exc
            raise

    # ------------------------------------------------------------------ #
    # Public query interface
    # ------------------------------------------------------------------ #
    def query(self, entity: str, **params: Any) -> Dict[str, Any]:
        """
        Hit a single page of the given endpoint. Use :meth:`stream` or
        :meth:`fetch_all` for bulk retrieval.
        """
        try:
            path = self.ENDPOINTS[entity]
        except KeyError as exc:
            raise ValueError(
                f"Unknown entity {entity!r}. Valid: {sorted(self.ENDPOINTS)}"
            ) from exc
        # Drop None values; they would be sent as the literal string "None".
        cleaned = {k: v for k, v in params.items() if v is not None}
        return self._request(path, params=cleaned)

    def stream(
        self,
        entity: str,
        page_size: int = DEFAULT_PAGE_SIZE,
        raise_on_first_page: bool = True,
        **params: Any,
    ) -> Generator[Any, None, None]:
        """
        Stream all matching items across pages.

        The generator yields the integer total count first, then yields each
        individual result dict.

        Parameters
        ----------
        raise_on_first_page : bool, default True
            If True (recommended), a failed first-page fetch re-raises the
            underlying ``RequestException``. Set to False to keep the legacy
            behavior of yielding 0 and returning silently.
        """
        params = {k: v for k, v in params.items() if v is not None}
        params["page_size"] = page_size
        page = 1

        try:
            params["page"] = page
            first = self.query(entity, **params)
        except RequestException as exc:
            if raise_on_first_page:
                raise
            print(f"[gwportal] first page fetch failed: {exc}", file=sys.stderr)
            yield 0
            return

        total = int(first.get("count", 0))
        num_pages = int(first.get("num_pages", 1))
        yield total
        if total == 0:
            return

        yield from first.get("results", [])

        for page in range(2, num_pages + 1):
            try:
                params["page"] = page
                resp = self.query(entity, **params)
            except RequestException as exc:
                print(
                    f"[gwportal] page {page} fetch failed: {exc}", file=sys.stderr
                )
                break
            rows: List[Dict[str, Any]] = resp.get("results", [])
            if not rows:
                break
            yield from rows

    def fetch_all(
        self,
        entity: str,
        page_size: int = DEFAULT_PAGE_SIZE,
        **params: Any,
    ) -> List[Dict[str, Any]]:
        """Materialize :meth:`stream` into a list (drops the leading count)."""
        it: Iterable[Any] = self.stream(entity, page_size=page_size, **params)
        gen = iter(it)
        try:
            next(gen)  # discard the total-count header
        except StopIteration:
            return []
        return list(gen)

    # ------------------------------------------------------------------ #
    # Health check
    # ------------------------------------------------------------------ #
    def ping(self, timeout: float = 5.0) -> bool:
        """Cheap reachability probe (asks for one tile)."""
        try:
            self._request("tiles/", params={"page_size": 1}, timeout=timeout)
            return True
        except RequestException:
            return False
