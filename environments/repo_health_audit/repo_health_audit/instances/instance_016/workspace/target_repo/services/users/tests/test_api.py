"""Tests for user API."""

import pytest


def test_create_user_endpoint():
    from services.users.api import create_user
    assert create_user is not None


def test_get_user_endpoint():
    from services.users.api import get_user
    assert get_user is not None


def test_list_users_endpoint():
    from services.users.api import list_users
    assert list_users is not None
