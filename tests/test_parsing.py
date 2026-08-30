import pytest

from mobile_crawler.parsing import extract_price, parse_screen_size

# The HTML parsers need bs4; the pure helpers above do not.
bs4 = pytest.importorskip("bs4", reason="beautifulsoup4 is needed for the HTML parsers")

from mobile_crawler.parsing import (  # noqa: E402
    parse_listing_links,
    parse_product_features,
    parse_product_name,
    parse_product_page,
)

LISTING_HTML = """
<html><body>
  <a class="product_img_link" href="https://shop.example/p/1">one</a>
  <a class="product_img_link" href="https://shop.example/p/2">two</a>
  <a class="product_img_link" href="https://shop.example/p/1">dup</a>
  <a class="other_link" href="https://shop.example/not-a-product">no</a>
</body></html>
"""

PRODUCT_HTML = """
<html><body>
  <div class="rte align_justify"><h2>Galaxy Example 12</h2></div>
  <p class="our_price_display">12,500,000 تومان</p>
  <div role="tabpanel">
    <div class="features">
      <span class="feature_name">اندازه</span><span class="feature_value">6.5 اینچ</span>
    </div>
    <div class="features">
      <span class="feature_name">بلوتوث</span><span class="feature_value">5.0</span>
    </div>
    <div class="features">
      <span class="feature_name">broken</span>
    </div>
  </div>
</body></html>
"""


class TestExtractPrice:
    def test_strips_separators_and_currency_text(self):
        assert extract_price("12,500,000 تومان") == 12_500_000

    def test_plain_number(self):
        assert extract_price("450000") == 450_000

    @pytest.mark.parametrize("value", [None, "", "   ", "no digits here"])
    def test_missing_or_digitless_returns_none(self, value):
        assert extract_price(value) is None

    def test_normalises_persian_digits(self):
        """str.isdigit() is True for these, but int() cannot parse them."""
        assert extract_price("۱۲۳۴۵") == 12345

    def test_normalises_arabic_indic_digits(self):
        assert extract_price("١٢٣٤٥") == 12345


class TestParseScreenSize:
    def test_parses_inches(self):
        assert parse_screen_size("6.5 اینچ") == 6.5

    def test_parses_integer_size(self):
        assert parse_screen_size("7 اینچ") == 7.0

    @pytest.mark.parametrize("value", [None, "", "unknown", "6.5 cm"])
    def test_unparseable_returns_none_rather_than_raising(self, value):
        """The notebook raised UnboundLocalError on the first non-matching row."""
        assert parse_screen_size(value) is None


class TestParseListingLinks:
    def test_returns_only_product_links(self):
        links = parse_listing_links(LISTING_HTML)
        assert "https://shop.example/not-a-product" not in links

    def test_deduplicates_preserving_order(self):
        assert parse_listing_links(LISTING_HTML) == [
            "https://shop.example/p/1",
            "https://shop.example/p/2",
        ]

    def test_empty_page_yields_no_links(self):
        assert parse_listing_links("<html></html>") == []


class TestParseProductPage:
    def test_extracts_name_and_price(self):
        record = parse_product_page(PRODUCT_HTML)
        assert record["Product"] == "Galaxy Example 12"
        assert record["Price"] == 12_500_000

    def test_extracts_feature_pairs(self):
        features = parse_product_features(PRODUCT_HTML)
        assert features["اندازه"] == "6.5 اینچ"
        assert features["بلوتوث"] == "5.0"

    def test_skips_rows_missing_a_value(self):
        """Zipping two separate lists silently mismatched pairs instead."""
        assert "broken" not in parse_product_features(PRODUCT_HTML)

    def test_missing_panel_gives_empty_features(self):
        assert parse_product_features("<html></html>") == {}

    def test_missing_name_block_returns_none(self):
        assert parse_product_name("<html></html>") is None

    def test_page_with_nothing_still_returns_a_record(self):
        """The original raised TypeError assigning into a None features dict."""
        record = parse_product_page("<html></html>")
        assert record == {"Price": None, "Product": None}
