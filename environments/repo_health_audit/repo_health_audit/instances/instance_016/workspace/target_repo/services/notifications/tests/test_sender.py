"""Tests for notification sender."""

import pytest


def test_send_notification():
    from services.notifications.sender import send_notification
    assert send_notification is not None
