"""HTML parsing for product listings and product pages.

Split out from the network layer so every parser can be exercised against
fixture HTML with no live site. The store's markup is the one thing most likely
to change, so it is also the part most worth testing.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "extract_price",
    "parse_screen_size",
    "parse_listing_links",
    "parse_product_name",
    "parse_product_features",
    "parse_product_page",
]

#: Screen size is published as e.g. "6.5 اینچ" ("6.5 inch").
_SCREEN_SIZE = re.compile(r"^\s*([\d.]+)\s*اینچ")

#: Digits used by the store: ASCII, plus Persian and Arabic-Indic forms, which
#: the original character-by-character ``str.isdigit()`` loop silently dropped.
_DIGIT_TRANSLATION = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def extract_price(text: str | None) -> int | None:
    """Pull an integer price out of a price string.

    Returns ``None`` when ``text`` is missing or holds no digits, rather than
    raising - a product with no listed price is normal, not an error.

    Persian and Arabic-Indic digits are normalised to ASCII first; the original
    implementation used ``str.isdigit()``, which is True for those characters
    but produced a string ``int()`` could not parse.
    """
    if not text:
        return None
    digits = "".join(ch for ch in text.translate(_DIGIT_TRANSLATION) if ch.isascii() and ch.isdigit())
    return int(digits) if digits else None


def parse_screen_size(text: str | None) -> float | None:
    """Parse a screen size in inches from e.g. ``"6.5 اینچ"``.

    Returns ``None`` when the value is absent or unparseable. The notebook's
    version raised ``UnboundLocalError`` on the first non-matching row, because
    it appended a ``found`` variable that was only assigned inside the match.
    """
    if not text:
        return None
    match = _SCREEN_SIZE.match(str(text))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _soup(html: str):
    from bs4 import BeautifulSoup  # noqa: PLC0415

    return BeautifulSoup(html, "html.parser")


def parse_listing_links(html: str) -> list[str]:
    """Return product URLs from a category listing page.

    Duplicates are removed while preserving order, since the same product can
    appear on more than one listing page.
    """
    seen: dict[str, None] = {}
    for anchor in _soup(html).find_all("a", {"class": "product_img_link"}):
        href = anchor.get("href")
        if href:
            seen.setdefault(href, None)
    return list(seen)


def parse_product_name(html: str) -> str | None:
    """Return the product name, or ``None`` when the block is absent."""
    block = _soup(html).find("div", {"class": "rte align_justify"})
    if block is None:
        return None
    heading = block.find("h2")
    if heading is None:
        return None
    return heading.get_text(strip=True) or None


def parse_product_features(html: str) -> dict[str, str]:
    """Return the product's specification table as a name -> value mapping.

    Returns an empty dict when the panel is missing. The original code paired
    names and values by building two separate lists and zipping them, which
    silently mismatched whenever a row was missing one half; here each row is
    read as a unit and skipped if incomplete.
    """
    panel = _soup(html).find("div", {"role": "tabpanel"})
    if panel is None:
        return {}

    features: dict[str, str] = {}
    for row in panel.find_all("div", {"class": "features"}):
        name_tag = row.find("span", {"class": "feature_name"})
        value_tag = row.find("span", {"class": "feature_value"})
        if name_tag is None or value_tag is None:
            continue
        name = name_tag.get_text(strip=True)
        if name:
            features[name] = value_tag.get_text(strip=True)
    return features


def parse_product_page(html: str) -> dict[str, Any]:
    """Parse one product page into a flat record.

    Always returns a dict containing at least ``Product`` and ``Price`` keys.
    The original returned ``None`` for the feature dict when the panel was
    missing and then immediately did ``result["Price"] = price``, raising
    ``TypeError: 'NoneType' object does not support item assignment``.
    """
    record: dict[str, Any] = dict(parse_product_features(html))

    price_node = _soup(html).find("p", {"class": "our_price_display"})
    record["Price"] = extract_price(price_node.get_text() if price_node else None)
    record["Product"] = parse_product_name(html)
    return record
