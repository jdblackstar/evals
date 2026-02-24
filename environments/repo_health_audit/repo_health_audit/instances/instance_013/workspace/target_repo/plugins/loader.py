from __future__ import annotations

import importlib
import pkgutil

import pluggy

from plugins.base import BasePlugin, PluginError

hookspec = pluggy.HookspecMarker("plugins")
hookimpl = pluggy.HookimplMarker("plugins")


def discover_plugins(package_path: str) -> list[type[BasePlugin]]:
    """Walk a package path and return all BasePlugin subclasses found."""
    found: list[type[BasePlugin]] = []
    try:
        pkg = importlib.import_module(package_path)
    except ModuleNotFoundError as exc:
        raise PluginError(f"could not import {package_path}") from exc

    for _importer, modname, _ispkg in pkgutil.walk_packages(
        pkg.__path__, prefix=pkg.__name__ + "."
    ):
        mod = importlib.import_module(modname)
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if (
                isinstance(obj, type)
                and issubclass(obj, BasePlugin)
                and obj is not BasePlugin
            ):
                found.append(obj)
    return found
