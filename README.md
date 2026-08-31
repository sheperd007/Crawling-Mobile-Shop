# Crawling Mobile Shop

Scrape mobile-phone listings from an Iranian e-commerce site, engineer their technical specifications into model-ready features, and fit a linear regression to predict price from hardware.

**Tech stack:** Python · requests · BeautifulSoup · pandas · scikit-learn · statsmodels

---

## Overview

Mobile-phone prices are largely driven by hardware specifications (chip, GPU, screen, connectivity, build, release year). This project collects those specifications directly from product pages and tests how well a linear model recovers price from them.

The pipeline has three stages:

1. **Crawl** — walk the store's mobile-category listing, follow each product link, and parse the per-product specification table.
2. **Prepare** — map the Persian specification names to English model columns, parse screen size to numeric inches, and one-hot encode the categorical specs.
3. **Model** — fit a `LinearRegression` with a held-out test split, and optionally a `statsmodels` OLS for inference.

## What's inside

```
Crawling-Mobile-Shop/
├── src/mobile_crawler/
│   ├── config.py       # Base URL, page range, headers, timeouts, retries
│   ├── parsing.py      # HTML -> records. Pure; testable against fixtures
│   ├── crawler.py      # HTTP layer: listing walk, product fetch, retries
│   ├── features.py     # Persian -> English mapping, design matrix
│   ├── modeling.py     # Fit, evaluate, OLS summary
│   └── cli.py          # crawl / train
├── tests/              # 61 tests; no network, no live site
├── legacy/
│   └── crawler_original.py      # Original script, preserved
├── pyproject.toml
└── requirements.txt
```

Parsing and feature preparation are separated from the network layer, so the part most likely to break — the store's markup — is the part covered by tests, using fixture HTML rather than live requests.

## Installation

```bash
git clone https://github.com/Hamid-Jahani/Crawling-Mobile-Shop.git
cd Crawling-Mobile-Shop
pip install -r requirements.txt
pip install -e .
```

## Usage

### Command line

```bash
# Scrape to CSV (pages 1-7 by default)
mobile-crawler crawl --output mobiles.csv --verbose

# Fit the price model from the scraped CSV
mobile-crawler train --input mobiles.csv

# Add the statsmodels OLS summary
mobile-crawler train --input mobiles.csv --ols
```

### Python

```python
from mobile_crawler import CrawlConfig, crawl, select_features, build_design_matrix
from mobile_crawler.modeling import fit_price_model

raw = crawl(CrawlConfig(last_page=3))
features = select_features(raw)
X, y = build_design_matrix(features)

result = fit_price_model(X, y)
print(result)                        # train/test R^2 and RMSE
print(result.coefficients.head(10))  # ranked by magnitude
```

## Fixes applied during the refactor

The original `crawler.py` is preserved in `legacy/`; the exploration notebook is no longer tracked and remains in the git history. Porting them surfaced several defects, all fixed here:

**Feature columns were mislabelled.** The notebook selected thirteen Persian columns in one order and then assigned English names in a *different* order:

```python
Data_machine = Data[['Price','Product','Wi-Fi','اندازه','تراکم پیکسلی','بلوتوث', ...]]
Data_machine.columns = ["Price","Product","WIFI","bluetooth","Pixel_density","size", ...]
```

Position 4 is `اندازه` (**size**) but received the label `bluetooth`; position 6 is `بلوتوث` (**bluetooth**) but received `size`. Eight of thirteen columns were affected, including a CPU/GPU swap, so every regression coefficient was attributed to the wrong feature. `features.COLUMN_MAP` now maps by name, and tests pin each corrected pairing.

**The crawler could not run on modern pandas.** It accumulated rows with `DataFrame.append`, removed in pandas 2.0. Rows are now collected in a list and passed to the constructor once — which also removes the quadratic reallocation of appending inside the loop.

**A missing specification panel crashed the parser.** The feature dict was set to `None` in an `except` branch, then immediately assigned into (`result["Price"] = price`), raising `TypeError`. Parsing now always returns a dict.

**Screen-size parsing raised on the first non-matching row.** `found` was assigned only inside the regex match, so a row without "اینچ" hit `UnboundLocalError`. `parse_screen_size` returns `None` instead.

**Prices with Persian digits were silently corrupted.** The digit filter used `str.isdigit()`, which is `True` for `۰-۹`, producing a string `int()` could not parse. Digits are normalised to ASCII first.

**Requests had no timeout and no retry**, so one stalled response hung the crawl and one transient error silently lost a product. Both are now configurable, with a polite delay between requests.

**Name/value pairs could mismatch.** Feature names and values were collected into two separate lists and zipped, so a row missing one half shifted every subsequent pairing. Each row is now read as a unit and skipped if incomplete.

## Development

```bash
pip install -e ".[dev]"
pytest
```

61 tests cover price and screen-size parsing (including Persian and Arabic-Indic digits), listing and product-page parsing against fixture HTML, the column mapping, design-matrix construction, config validation, and CLI parsing. None touch the network.

The HTML parser tests require `beautifulsoup4`; without it they skip rather than fail (38 pass, 1 module skips). `crawler.py`'s HTTP paths are not automatically tested, since exercising them requires the live site.

## Notes

The crawler depends on the live structure of the target site. If its markup or pagination changes, the selectors in `parsing.py` need updating — the fixture-based tests will keep passing, since they pin the parser's contract rather than the site's current HTML.

## License

Released under the [MIT License](LICENSE).
