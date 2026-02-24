import subprocess


def run_engine(input_path: str, output_path: str) -> int:
    result = subprocess.run(
        ["./engine/target/release/engine", input_path, output_path],
        capture_output=True,
    )
    return result.returncode


def validate_output(output_path: str) -> bool:
    try:
        with open(output_path) as f:
            data = f.read()
        return len(data) > 0
    except FileNotFoundError:
        return False
