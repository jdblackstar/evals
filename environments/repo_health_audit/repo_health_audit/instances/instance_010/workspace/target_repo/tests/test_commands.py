from click.testing import CliRunner

from cli.main import cli


def test_greet_prints_name() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["greet", "Alice"])
    assert result.exit_code == 0
    assert "Alice" in result.output


def test_greet_unknown_name() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["greet", "World"])
    assert result.exit_code == 0
    assert "World" in result.output


def test_process_missing_file() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["process", "nonexistent.txt"])
    assert result.exit_code != 0


def test_process_valid_file(tmp_path) -> None:
    p = tmp_path / "data.txt"
    p.write_text("line1\nline2\nline3\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["process", str(p)])
    assert result.exit_code == 0
    assert "3 lines" in result.output
