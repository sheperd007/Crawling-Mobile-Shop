"""Feature selection and preparation for the price model.

The notebook selected thirteen Persian-named columns positionally and then
overwrote ``DataFrame.columns`` with a list of English names in a *different*
order, so eight of thirteen features carried the wrong label and every
regression coefficient was attributed to the wrong variable.

This module maps names explicitly, so a reordering cannot silently mislabel
anything. See ``COLUMN_MAP`` and the tests that pin it.
"""

from __future__ import annotations

import pandas as pd

from .parsing import parse_screen_size

__all__ = ["COLUMN_MAP", "SCREEN_SIZE_COLUMN", "select_features", "build_design_matrix"]

#: Persian source column -> English model column.
#:
#: Correcting the notebook's positional assignment. Verified pairings:
#:   اندازه              = size            (notebook labelled it "bluetooth")
#:   بلوتوث              = bluetooth       (notebook labelled it "size")
#:   تراشه               = chip            (notebook labelled it "resolution")
#:   جنس بدنه            = body material   (notebook labelled it "year")
#:   سال تولید           = production year (notebook labelled it "Body_Material")
#:   رزولوشن             = resolution      (notebook labelled it "Chip")
#:   پردازنده گرافیکی    = GPU             (notebook labelled it "CPU")
#:   پردازنده مرکزی      = CPU             (notebook labelled it "GPU")
COLUMN_MAP: dict[str, str] = {
    "Price": "price",
    "Product": "product",
    "Wi-Fi": "wifi",
    "اندازه": "screen_size_inches",
    "تراکم پیکسلی": "pixel_density",
    "بلوتوث": "bluetooth",
    "تراشه": "chip",
    "جنس بدنه": "body_material",
    "تعداد سیم کارت": "sim_card_count",
    "سال تولید": "production_year",
    "رزولوشن": "resolution",
    "پردازنده گرافیکی": "gpu",
    "پردازنده مرکزی": "cpu",
}

#: The one numeric feature, published as text like "6.5 اینچ".
SCREEN_SIZE_COLUMN = "screen_size_inches"

#: Identifier and target columns, excluded from the design matrix.
_NON_FEATURE_COLUMNS = ("price", "product")


def select_features(frame: pd.DataFrame, *, strict: bool = False) -> pd.DataFrame:
    """Select the modelling columns and rename them to English.

    Columns absent from ``frame`` are skipped unless ``strict`` is set, in
    which case a ``KeyError`` names every missing column at once. Scraped data
    is uneven, so tolerating gaps is the sensible default.

    The screen-size column is converted from text to float inches.
    """
    missing = [source for source in COLUMN_MAP if source not in frame.columns]
    if missing and strict:
        raise KeyError(f"missing expected column(s): {', '.join(missing)}")

    present = {source: target for source, target in COLUMN_MAP.items() if source in frame.columns}
    selected = frame[list(present)].rename(columns=present)

    if SCREEN_SIZE_COLUMN in selected.columns:
        selected[SCREEN_SIZE_COLUMN] = selected[SCREEN_SIZE_COLUMN].map(parse_screen_size)

    return selected


def build_design_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a selected frame into one-hot encoded ``(X, y)``.

    ``drop_first=True`` avoids the dummy-variable trap, which matters here
    because the model is fitted by ordinary least squares and a full set of
    dummies makes the design matrix singular.

    Raises ``KeyError`` if the target column is absent and ``ValueError`` if no
    rows survive dropping missing prices.
    """
    if "price" not in frame.columns:
        raise KeyError("frame has no 'price' column to use as the target")

    usable = frame.dropna(subset=["price"])
    if usable.empty:
        raise ValueError("no rows with a known price; nothing to model")

    target = usable["price"]
    predictors = usable.drop(columns=[c for c in _NON_FEATURE_COLUMNS if c in usable.columns])
    return pd.get_dummies(predictors, drop_first=True), target
