from tempfile import NamedTemporaryFile

import nox
from nox.sessions import Session

LOCATIONS = "src", "tests", "noxfile.py"


def install_with_constraints(session: Session, *args: str) -> None:
    with NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            f"--output={requirements.name}",
            external=True,
        )
        session.install(f"--constraint={requirements.name}", *args)


@nox.session(python=["3.8"])
def tests(session: Session) -> None:
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(session, "coverage[toml]", "pytest", "pytest-cov")
    session.run("pytest", "--cov")


@nox.session(python=["3.8"])
def lint(session: Session) -> None:
    args = session.posargs or LOCATIONS
    install_with_constraints(
        session, "flake8", "flake8-black", "flake8-isort", "flake8-annotations",
    )
    session.run("flake8", *args)
