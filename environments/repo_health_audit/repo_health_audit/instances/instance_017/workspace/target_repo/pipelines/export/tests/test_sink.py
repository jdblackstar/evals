"""Tests for export sink connectors."""

import pytest


def test_write_sink_local(tmp_path):
    from pipelines.export.sink import write_sink
    import pandas as pd

    df = pd.DataFrame({"id": ["1"], "value": [42.0]})
    path = str(tmp_path / "output.parquet")
    count = write_sink(df, "local", path)
    assert count == 1


def test_write_sink_unsupported_raises():
    from pipelines.export.sink import write_sink
    import pandas as pd

    df = pd.DataFrame({"id": ["1"]})
    with pytest.raises(ValueError, match="Unsupported sink type"):
        write_sink(df, "hdfs", "/fake")
