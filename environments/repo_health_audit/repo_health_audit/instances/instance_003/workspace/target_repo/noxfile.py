import nox


@nox.session
def monorepo_tests(session):
    session.run("python", "-m", "pytest", "packages", "integration", "-q")


@nox.session
def coverage_gate(session):
    session.run(
        "python",
        "-m",
        "pytest",
        "packages",
        "integration",
        "--cov=packages",
        "--cov=libs",
        "--cov-report=term",
    )
