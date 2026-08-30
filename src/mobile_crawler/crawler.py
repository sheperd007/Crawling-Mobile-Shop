"""Network layer: walk the listing pages and fetch each product.

Parsing lives in :mod:`mobile_crawler.parsing`; this module only deals with
HTTP and with assembling records into a DataFrame.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Iterable

import pandas as pd

from .config import CrawlConfig
from .parsing import parse_listing_links, parse_product_page

__all__ = ["fetch", "collect_product_links", "crawl"]

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


def fetch(url: str, config: CrawlConfig) -> str | None:
    """GET ``url`` and return its body, or ``None`` if every attempt fails.

    Retries with linear backoff. The original code had no timeout and no retry,
    so one slow response could hang the whole crawl and one transient error
    lost that product silently.
    """
    import requests  # noqa: PLC0415

    for attempt in range(config.retries + 1):
        try:
            response = requests.get(url, headers=config.headers, timeout=config.timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            if attempt == config.retries:
                logger.warning("giving up on %s after %d attempts: %s", url, attempt + 1, exc)
                return None
            time.sleep(config.retry_backoff * (attempt + 1))
    return None


def collect_product_links(config: CrawlConfig | None = None) -> list[str]:
    """Walk the listing pages and return every distinct product URL."""
    config = config or CrawlConfig()
    links: dict[str, None] = {}

    for page in config.pages:
        html = fetch(config.page_url(page), config)
        if html is None:
            continue
        found = parse_listing_links(html)
        logger.info("page %d: %d product links", page, len(found))
        for link in found:
            links.setdefault(link, None)

    return list(links)


def crawl(
    config: CrawlConfig | None = None,
    *,
    links: Iterable[str] | None = None,
    on_progress: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Crawl the store and return one row per product.

    Pass ``links`` to re-crawl a known set of URLs without re-walking the
    listing pages.

    Rows are collected in a list and handed to the DataFrame constructor once.
    The original appended to a DataFrame inside the loop via
    ``DataFrame.append``, which was removed in pandas 2.0 - so that code raises
    ``AttributeError`` on any current pandas - and was quadratic besides,
    reallocating the whole frame on every product.
    """
    config = config or CrawlConfig()
    urls = list(links) if links is not None else collect_product_links(config)

    records: list[dict] = []
    for index, url in enumerate(urls):
        if on_progress is not None:
            on_progress(index, len(urls), url)

        html = fetch(url, config)
        if html is None:
            continue

        record = parse_product_page(html)
        record["url"] = url
        records.append(record)

        if config.polite_delay:
            time.sleep(config.polite_delay)

    logger.info("collected %d of %d products", len(records), len(urls))
    return pd.DataFrame(records)
