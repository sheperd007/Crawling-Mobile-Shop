"""Crawler configuration.

The original script hard-coded the base URL, the page range, and a browser
User-Agent inline, and issued requests with no timeout. Those choices are
gathered here so a caller can adjust them without editing crawl logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Listing page for the store's mobile category.
DEFAULT_BASE_URL = (
    "https://kalatik.com/21-mobile"
    "#/%D9%85%D9%88%D8%AC%D9%88%D8%AF_%D8%A8%D9%88%D8%AF%D9%86"
    "-%D9%85%D9%88%D8%AC%D9%88%D8%AF%DB%8C"
)

#: Some stores serve a reduced page to non-browser agents, so a browser
#: User-Agent is sent on every request rather than only on product pages.
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
    )
}


@dataclass(frozen=True)
class CrawlConfig:
    """Settings for one crawl run."""

    base_url: str = DEFAULT_BASE_URL
    first_page: int = 1
    last_page: int = 7
    """Inclusive. The original ``range(1, 8)`` covered pages 1-7."""

    headers: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_HEADERS))

    timeout: float = 30.0
    """Seconds. The original code passed no timeout, so a stalled server could
    hang the crawl indefinitely."""

    retries: int = 2
    retry_backoff: float = 1.0
    polite_delay: float = 0.5
    """Seconds between product requests, to avoid hammering the store."""

    def __post_init__(self) -> None:
        if self.first_page < 1:
            raise ValueError(f"first_page must be >= 1, got {self.first_page}")
        if self.last_page < self.first_page:
            raise ValueError(
                f"last_page ({self.last_page}) must be >= first_page ({self.first_page})"
            )
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
        if self.retries < 0:
            raise ValueError(f"retries must be >= 0, got {self.retries}")

    @property
    def pages(self) -> range:
        """Page numbers to visit, as an inclusive range."""
        return range(self.first_page, self.last_page + 1)

    def page_url(self, page: int) -> str:
        """URL of a given listing page."""
        return f"https://kalatik.com/21-mobile#/%D9%85%D9%88%D8%AC%D9%88%D8%AF_%D8%A8%D9%88%D8%AF%D9%86-%D9%85%D9%88%D8%AC%D9%88%D8%AF%DB%8C/page-{page}"
