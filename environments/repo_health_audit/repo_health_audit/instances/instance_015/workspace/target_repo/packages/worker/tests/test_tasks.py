"""Tests for worker tasks."""

import pytest


def test_process_item():
    from packages.worker.src.tasks import process_item
    assert process_item is not None


def test_enqueue_processing():
    from packages.worker.src.tasks import enqueue_processing
    assert enqueue_processing is not None
