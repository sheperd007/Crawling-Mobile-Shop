"""Command-line interface.

    mobile-crawler crawl --output mobiles.csv
    mobile-crawler train --input mobiles.csv [--ols]

Crawling and modelling are separate commands so the scrape is done once and
the model can be refitted from the CSV without touching the network again.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import CrawlConfig

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mobile-crawler",
        description="Scrape mobile-phone specifications and model price from hardware.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress")
    sub = parser.add_subparsers(dest="command", required=True)

    crawl = sub.add_parser("crawl", help="scrape product listings to CSV")
    crawl.add_argument("--output", type=Path, default=Path("mobiles.csv"))
    crawl.add_argument("--first-page", type=int, default=CrawlConfig().first_page)
    crawl.add_argument("--last-page", type=int, default=CrawlConfig().last_page)
    crawl.add_argument("--timeout", type=float, default=CrawlConfig().timeout)
    crawl.add_argument("--delay", type=float, default=CrawlConfig().polite_delay,
                       help="seconds to wait between product requests")

    train = sub.add_parser("train", help="fit the price model from a scraped CSV")
    train.add_argument("--input", type=Path, required=True)
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--random-state", type=int, default=101)
    train.add_argument("--ols", action="store_true", help="also print the statsmodels OLS summary")
    train.add_argument("--top", type=int, default=15, help="how many coefficients to show")

    return parser


def _cmd_crawl(args: argparse.Namespace) -> int:
    from .crawler import crawl

    config = CrawlConfig(
        first_page=args.first_page,
        last_page=args.last_page,
        timeout=args.timeout,
        polite_delay=args.delay,
    )

    def report(index: int, total: int, url: str) -> None:
        print(f"  [{index + 1}/{total}] {url}", file=sys.stderr)

    frame = crawl(config, on_progress=report if args.verbose else None)
    if frame.empty:
        print("error: no products collected", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"wrote {len(frame)} rows x {len(frame.columns)} columns -> {args.output}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    import pandas as pd

    from .features import build_design_matrix, select_features
    from .modeling import fit_price_model, ols_summary

    frame = pd.read_csv(args.input)
    selected = select_features(frame)
    predictors, target = build_design_matrix(selected)

    print(f"design matrix: {predictors.shape[0]} rows x {predictors.shape[1]} columns")
    if predictors.shape[1] >= predictors.shape[0]:
        print(
            "warning: at least as many columns as rows after one-hot encoding; "
            "coefficients will not be identifiable",
            file=sys.stderr,
        )

    result = fit_price_model(
        predictors, target, test_size=args.test_size, random_state=args.random_state
    )
    print(result)
    print(f"\nintercept: {result.intercept:,.0f}")
    print(f"\ntop {args.top} coefficients by magnitude:")
    print(result.coefficients.head(args.top).to_string())

    if args.ols:
        print("\n" + ols_summary(predictors, target))
    return 0


_COMMANDS = {"crawl": _cmd_crawl, "train": _cmd_train}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    try:
        return _COMMANDS[args.command](args)
    except (FileNotFoundError, KeyError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
