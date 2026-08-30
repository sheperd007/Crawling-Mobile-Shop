import pandas as pd
import pytest

from mobile_crawler.features import (
    COLUMN_MAP,
    SCREEN_SIZE_COLUMN,
    build_design_matrix,
    select_features,
)


@pytest.fixture
def raw():
    """A scraped frame using the store's Persian column names."""
    return pd.DataFrame(
        {
            "Price": [10_000_000, 20_000_000, 15_000_000, 30_000_000, 25_000_000, 12_000_000],
            "Product": ["a", "b", "c", "d", "e", "f"],
            "Wi-Fi": ["yes", "yes", "no", "yes", "no", "yes"],
            "اندازه": ["6.5 اینچ", "6.1 اینچ", "5.8 اینچ", "6.7 اینچ", "6.0 اینچ", "6.4 اینچ"],
            "بلوتوث": ["5.0", "5.1", "4.2", "5.2", "5.0", "5.1"],
            "سال تولید": ["2021", "2022", "2020", "2022", "2021", "2022"],
            "پردازنده مرکزی": ["cpu1", "cpu2", "cpu1", "cpu2", "cpu1", "cpu2"],
            "پردازنده گرافیکی": ["gpu1", "gpu2", "gpu1", "gpu2", "gpu1", "gpu2"],
        }
    )


class TestColumnMap:
    """Pins the fix for the notebook's positional column assignment.

    The notebook selected Persian columns in one order and assigned English
    names in another, mislabelling eight of thirteen features. These assertions
    fail if that pairing is ever reintroduced.
    """

    def test_screen_size_is_not_labelled_bluetooth(self):
        assert COLUMN_MAP["اندازه"] == "screen_size_inches"

    def test_bluetooth_is_not_labelled_size(self):
        assert COLUMN_MAP["بلوتوث"] == "bluetooth"

    def test_cpu_and_gpu_are_not_swapped(self):
        assert COLUMN_MAP["پردازنده مرکزی"] == "cpu"
        assert COLUMN_MAP["پردازنده گرافیکی"] == "gpu"

    def test_year_and_body_material_are_not_swapped(self):
        assert COLUMN_MAP["سال تولید"] == "production_year"
        assert COLUMN_MAP["جنس بدنه"] == "body_material"

    def test_chip_and_resolution_are_not_swapped(self):
        assert COLUMN_MAP["تراشه"] == "chip"
        assert COLUMN_MAP["رزولوشن"] == "resolution"

    def test_every_target_name_is_unique(self):
        targets = list(COLUMN_MAP.values())
        assert len(targets) == len(set(targets))


class TestSelectFeatures:
    def test_renames_to_english(self, raw):
        selected = select_features(raw)
        assert "price" in selected.columns
        assert "screen_size_inches" in selected.columns
        assert "اندازه" not in selected.columns

    def test_screen_size_becomes_numeric_inches(self, raw):
        selected = select_features(raw)
        assert selected[SCREEN_SIZE_COLUMN].tolist()[:3] == [6.5, 6.1, 5.8]

    def test_absent_columns_are_skipped_by_default(self, raw):
        selected = select_features(raw.drop(columns=["بلوتوث"]))
        assert "bluetooth" not in selected.columns
        assert "screen_size_inches" in selected.columns

    def test_strict_mode_names_every_missing_column(self, raw):
        with pytest.raises(KeyError) as excinfo:
            select_features(raw, strict=True)
        message = str(excinfo.value)
        assert "تراشه" in message and "رزولوشن" in message

    def test_does_not_mutate_the_input(self, raw):
        before = raw.copy()
        select_features(raw)
        pd.testing.assert_frame_equal(raw, before)


class TestBuildDesignMatrix:
    def test_target_is_price(self, raw):
        _, target = build_design_matrix(select_features(raw))
        assert target.name == "price"

    def test_identifier_and_target_are_not_predictors(self, raw):
        predictors, _ = build_design_matrix(select_features(raw))
        assert "price" not in predictors.columns
        assert "product" not in predictors.columns

    def test_categoricals_are_one_hot_encoded(self, raw):
        predictors, _ = build_design_matrix(select_features(raw))
        assert any(c.startswith("cpu_") for c in predictors.columns)

    def test_drop_first_avoids_the_dummy_variable_trap(self, raw):
        """A binary column must yield one dummy, not two."""
        predictors, _ = build_design_matrix(select_features(raw))
        assert len([c for c in predictors.columns if c.startswith("cpu_")]) == 1

    def test_rows_without_a_price_are_dropped(self, raw):
        raw.loc[0, "Price"] = None
        predictors, target = build_design_matrix(select_features(raw))
        assert len(target) == len(raw) - 1
        assert len(predictors) == len(target)

    def test_missing_target_column_raises(self):
        with pytest.raises(KeyError, match="price"):
            build_design_matrix(pd.DataFrame({"other": [1, 2]}))

    def test_all_prices_missing_raises(self, raw):
        raw["Price"] = None
        with pytest.raises(ValueError, match="no rows with a known price"):
            build_design_matrix(select_features(raw))
