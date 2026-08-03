import pytest

from gemma_hate_inference.cli import build_parser
from gemma_hate_inference.platforms import SUPPORTED_PLATFORMS


def test_run_accepts_new_messaging_platforms():
    parser = build_parser()

    args = parser.parse_args(
        ["run", "--platform", "WhatsApp", "--platform", "Telegram"]
    )

    assert tuple(args.platforms) == ("whatsapp", "telegram")


def test_supported_platforms_include_messaging_sources():
    assert "whatsapp" in SUPPORTED_PLATFORMS
    assert "telegram" in SUPPORTED_PLATFORMS


def test_run_rejects_unknown_platform():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--platform", "unknown"])
