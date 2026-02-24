"""Tests for gateway proxy."""

import pytest


def test_forward_request():
    from services.gateway.proxy import forward_request
    assert forward_request is not None


def test_service_registry():
    from services.gateway.proxy import SERVICE_REGISTRY
    assert "users" in SERVICE_REGISTRY
    assert "notifications" in SERVICE_REGISTRY
