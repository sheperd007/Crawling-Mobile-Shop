# Crawling Mobile Shop

Scrape mobile-phone listings from an Iranian e-commerce site, engineer their technical specifications into model-ready features, and fit a linear regression to predict price from hardware.

**Tech stack:** Python · requests · BeautifulSoup · pandas · NumPy · scikit-learn · statsmodels · seaborn · pandas-profiling

---

## Overview

Mobile-phone prices are largely driven by hardware specifications (chip, GPU, screen, connectivity, build, release year, etc.). This project collects those specifications directly from product pages on an online store and tests how well a simple linear model can recover price from them.

The pipeline has two stages:

1. **Crawling** — `crawler.py` walks the store's mobile-category listing, follows each product link, and parses the per-product specification table into a tidy `pandas` DataFrame.
2. **Feature engineering & modeling** — the notebook cleans the raw specs, selects and renames a subset of features to English, one-hot encodes the categorical fields, and fits both a scikit-learn `LinearRegression` and a `statsmodels` OLS regression for interpretability.

The raw crawl produces **140 records across 69 raw feature columns**.

## What's inside

| Path | Description |
|------|-------------|
| `crawler.py` | Scraper. Defines `get_data()`, which paginates the listing, follows product links, extracts name, price, and the full feature table per phone, and returns a `pandas.DataFrame`. |
| `Mobile Crawling .ipynb` | End-to-end notebook: runs the crawler, cleans features, builds the modeling matrix, and fits the regression models. |
| `LICENSE` | MIT license. |
| `.gitignore` | Standard ignores. |

## Methods / Approach

**Scraping (`crawler.py`)**
- Iterates listing pages and collects product URLs via the `product_img_link` anchors with BeautifulSoup.
- For each product, parses the name, the displayed price (digits extracted and cast to `int`), and the key/value specification table (`feature_name` / `feature_value` spans).
- Assembles all phones into a single DataFrame, with each distinct specification becoming a column.

**Feature engineering (notebook)**
- Cleans noisy fields with regex (e.g. extracting the numeric screen size from its raw string).
- Selects a focused subset of specifications — Wi-Fi, screen size, pixel density, Bluetooth, chip, body material, SIM-card count, release year, resolution, GPU, CPU — and renames them to English.
- Generates a `pandas-profiling` report for exploratory data analysis.
- One-hot encodes the categorical specifications with `pd.get_dummies(drop_first=True)`, yielding the final model matrix.

**Modeling (notebook)**
- Splits the data into train/test sets (`test_size=0.2`, `random_state=101`) → 112 train / 28 test rows over 66 encoded features.
- Fits a scikit-learn `LinearRegression`, then refits an OLS model with `statsmodels` to inspect coefficients, significance, and overall fit.

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the crawler programmatically
python -c "from crawler import get_data; df = get_data(); print(df.shape); df.to_csv('mobiles.csv', index=False)"
```

Or open the notebook to reproduce the full pipeline end to end:

```bash
jupyter notebook "Mobile Crawling .ipynb"
```

> Note: the crawler depends on the live structure of the target site. If the site's markup or pagination changes, the CSS selectors in `crawler.py` may need updating.

## Results

The `statsmodels` OLS fit on the training data reports a high in-sample fit (**R² ≈ 0.96, Adjusted R² ≈ 0.95**). The model summary also flags a near-singular design matrix and strong multicollinearity — expected when many high-cardinality categorical specs are one-hot encoded on a small (140-row) sample. These figures should therefore be read as in-sample fit on this dataset rather than a validated out-of-sample accuracy claim; the project is best viewed as a scraping-plus-feature-engineering pipeline and a baseline modeling exploration.

## License

Released under the [MIT License](LICENSE).
