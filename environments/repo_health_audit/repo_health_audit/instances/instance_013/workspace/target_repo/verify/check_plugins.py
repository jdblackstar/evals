from plugins.base import BasePlugin, PluginError
from plugins.builtins.csv_plugin import CsvPlugin
from plugins.builtins.json_plugin import JsonPlugin


def test_csv_plugin_parses_valid_csv():
    plugin = CsvPlugin()
    result = plugin.process(b"name,age\nAlice,30\nBob,25\n")
    assert result["format"] == "csv"
    assert result["row_count"] == 2


def test_json_plugin_parses_valid_json():
    plugin = JsonPlugin()
    result = plugin.process(b'{"key": "value"}')
    assert result["format"] == "json"
    assert result["type"] == "dict"


def test_base_plugin_rejects_empty_input():
    plugin = CsvPlugin()
    try:
        plugin.validate_input(b"")
        assert False, "expected PluginError"
    except PluginError:
        assert True


def test_json_plugin_rejects_invalid_json():
    plugin = JsonPlugin()
    try:
        plugin.process(b"not json at all")
        assert False, "expected PluginError"
    except PluginError:
        assert True
