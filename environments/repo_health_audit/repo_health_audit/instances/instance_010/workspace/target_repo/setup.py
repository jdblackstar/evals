from setuptools import setup, find_packages

setup(
    name="cli-tool",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
    install_requires=[
        "click>=8.1.0",
        "rich>=13.7.0",
        "toml>=0.10.2",
    ],
    entry_points={
        "console_scripts": [
            "cli-tool=cli.main:cli",
        ],
    },
)
