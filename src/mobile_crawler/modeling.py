"""Fitting and reporting the price regression.

Two views of the same model: scikit-learn for prediction and metrics, and
statsmodels OLS for the inferential summary (standard errors, t-tests, R^2).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = ["FitResult", "fit_price_model", "ols_summary"]


@dataclass
class FitResult:
    """A fitted model with its held-out performance and coefficients."""

    model: object
    coefficients: pd.DataFrame
    intercept: float
    r2_train: float
    r2_test: float
    rmse_test: float
    n_train: int
    n_test: int

    def __str__(self) -> str:
        return (
            f"train R^2={self.r2_train:.3f} (n={self.n_train})  "
            f"test R^2={self.r2_test:.3f} RMSE={self.rmse_test:,.0f} (n={self.n_test})"
        )


def fit_price_model(
    predictors: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 101,
) -> FitResult:
    """Fit a linear price model and evaluate it on a held-out split.

    ``random_state`` defaults to the notebook's value so results reproduce.

    Reports train *and* test R^2. The notebook reported neither, printing only
    the intercept and coefficients, which gives no signal about whether the
    model generalises - important here, because one-hot encoding a handful of
    scraped categorical specs can produce more columns than rows.
    """
    import numpy as np  # noqa: PLC0415
    from sklearn.linear_model import LinearRegression  # noqa: PLC0415
    from sklearn.metrics import mean_squared_error, r2_score  # noqa: PLC0415
    from sklearn.model_selection import train_test_split  # noqa: PLC0415

    if len(predictors) != len(target):
        raise ValueError(
            f"predictors and target differ in length: {len(predictors)} vs {len(target)}"
        )
    if len(predictors) < 5:
        raise ValueError(f"need at least 5 rows to fit and evaluate, got {len(predictors)}")

    x_train, x_test, y_train, y_test = train_test_split(
        predictors, target, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    coefficients = (
        pd.DataFrame({"coefficient": model.coef_}, index=predictors.columns)
        .sort_values("coefficient", key=abs, ascending=False)
    )

    return FitResult(
        model=model,
        coefficients=coefficients,
        intercept=float(model.intercept_),
        r2_train=float(r2_score(y_train, model.predict(x_train))),
        r2_test=float(r2_score(y_test, predictions)),
        rmse_test=float(np.sqrt(mean_squared_error(y_test, predictions))),
        n_train=len(x_train),
        n_test=len(x_test),
    )


def ols_summary(predictors: pd.DataFrame, target: pd.Series) -> str:
    """Return a statsmodels OLS summary for inference on the full sample.

    Fitted on all rows rather than the training split: this view is for
    interpreting coefficients, not for estimating generalisation.
    """
    import statsmodels.api as sm  # noqa: PLC0415

    design = sm.add_constant(predictors.astype(float))
    return str(sm.OLS(target.astype(float), design).fit().summary())
