from tempfile import NamedTemporaryFile

import nox

LOCATIONS = "src", "tests", "noxfile.py"


def install_with_constraints(session, *args, **kwargs):
    with NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            f"--output={requirements.name}",
            external=True,
        )
        session.install(f"--constraint={requirements.name}", *args, **kwargs)


@nox.session(python=["3.8"])
def tests(session):
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(session, "coverage[toml]", "pytest", "pytest-cov")
    session.run("pytest", "--cov")


@nox.session(python=["3.8"])
def lint(session):
    args = session.posargs or LOCATIONS
    install_with_constraints(session, "flake8", "flake8-black", "flake8-isort")
    session.run("flake8", *args)
