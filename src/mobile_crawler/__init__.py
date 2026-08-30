"""Scrape mobile-phone specifications and model price from hardware.

Typical use::

    from mobile_crawler import CrawlConfig, crawl, select_features, build_design_matrix
    from mobile_crawler.modeling import fit_price_model

    raw = crawl(CrawlConfig(last_page=3))
    features = select_features(raw)
    X, y = build_design_matrix(features)
    print(fit_price_model(X, y))

Parsing, feature preparation, and configuration import nothing heavier than
pandas, so they can be used and tested without network access.
"""

from __future__ import annotations

from .config import DEFAULT_BASE_URL, DEFAULT_HEADERS, CrawlConfig
from .features import COLUMN_MAP, build_design_matrix, select_features
from .parsing import (
    extract_price,
    parse_listing_links,
    parse_product_features,
    parse_product_name,
    parse_product_page,
    parse_screen_size,
)

__version__ = "0.1.0"

__all__ = [
    "COLUMN_MAP",
    "DEFAULT_BASE_URL",
    "DEFAULT_HEADERS",
    "CrawlConfig",
    "__version__",
    "build_design_matrix",
    "collect_product_links",
    "crawl",
    "extract_price",
    "fit_price_model",
    "ols_summary",
    "parse_listing_links",
    "parse_product_features",
    "parse_product_name",
    "parse_product_page",
    "parse_screen_size",
    "select_features",
]


def __getattr__(name: str):
    """Expose the network and modelling entry points without importing
    requests, scikit-learn, or statsmodels at package import time."""
    if name in {"crawl", "collect_product_links"}:
        from . import crawler

        return getattr(crawler, name)
    if name in {"fit_price_model", "ols_summary"}:
        from . import modeling

        return getattr(modeling, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
