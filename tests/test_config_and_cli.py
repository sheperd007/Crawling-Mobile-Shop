import pytest

from mobile_crawler.cli import build_parser
from mobile_crawler.config import CrawlConfig


class TestCrawlConfig:
    def test_defaults_match_the_original_script(self):
        config = CrawlConfig()
        assert config.first_page == 1
        assert config.last_page == 7  # the original range(1, 8)
        assert list(config.pages) == [1, 2, 3, 4, 5, 6, 7]

    def test_has_a_timeout_by_default(self):
        """The original passed no timeout, so a stalled server hung the crawl."""
        assert CrawlConfig().timeout > 0

    def test_sends_a_browser_user_agent(self):
        assert "Mozilla" in CrawlConfig().headers["User-Agent"]

    def test_page_url_includes_the_page_number(self):
        assert CrawlConfig().page_url(3).endswith("page-3")

    def test_page_range_is_inclusive(self):
        assert list(CrawlConfig(first_page=2, last_page=4).pages) == [2, 3, 4]

    def test_single_page_is_allowed(self):
        assert list(CrawlConfig(first_page=1, last_page=1).pages) == [1]

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"first_page": 0}, "first_page"),
            ({"first_page": 5, "last_page": 2}, "last_page"),
            ({"timeout": 0}, "timeout"),
            ({"timeout": -1}, "timeout"),
            ({"retries": -1}, "retries"),
        ],
    )
    def test_rejects_invalid_settings(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            CrawlConfig(**kwargs)

    def test_headers_are_not_shared_between_instances(self):
        """A mutable default would leak edits across configs."""
        first = CrawlConfig()
        first.headers["X-Test"] = "1"
        assert "X-Test" not in CrawlConfig().headers

    def test_is_immutable(self):
        with pytest.raises(Exception):
            CrawlConfig().timeout = 5


class TestCli:
    def test_a_command_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_crawl_defaults_match_the_config(self):
        args = build_parser().parse_args(["crawl"])
        assert args.first_page == CrawlConfig().first_page
        assert args.last_page == CrawlConfig().last_page
        assert args.timeout == CrawlConfig().timeout

    def test_train_requires_an_input(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["train"])

    def test_train_parses_flags(self):
        args = build_parser().parse_args(["train", "--input", "m.csv", "--ols"])
        assert args.ols is True
        assert args.random_state == 101  # the notebook's seed

    def test_unknown_command_is_rejected(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["scrape-everything"])

    @pytest.mark.parametrize("command", ["crawl", "train"])
    def test_every_subcommand_has_help(self, command, capsys):
        with pytest.raises(SystemExit) as excinfo:
            build_parser().parse_args([command, "--help"])
        assert excinfo.value.code == 0
        assert capsys.readouterr().out
