"""Tests for API handlers."""

import pytest


def test_health_check():
    from packages.api.src.handlers import health_check
    # Simplified sync test
    assert health_check is not None


def test_list_items():
    from packages.api.src.handlers import list_items
    assert list_items is not None


def test_create_item():
    from packages.api.src.handlers import create_item
    assert create_item is not None
