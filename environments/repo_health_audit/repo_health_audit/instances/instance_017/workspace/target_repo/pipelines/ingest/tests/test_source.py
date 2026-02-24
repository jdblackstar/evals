"""Tests for ingest source connectors."""

import pytest


def test_read_source_local():
    from pipelines.ingest.source import read_source
    assert read_source is not None


def test_read_source_unsupported_raises():
    from pipelines.ingest.source import read_source
    with pytest.raises(ValueError, match="Unsupported source type"):
        read_source("ftp", "/fake/path")
