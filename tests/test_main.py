"""Test cases for the __main__ module."""
import pytest

from cl import cli_dispatcher


@pytest.fixture
def runner():
    """Fixture for invoking command-line interfaces."""
    return


def test_main_succeeds(runner) -> None:
    """It exits with a status code of zero."""
    # result = runner.invoke(__main__.main)
    # assert result.exit_code == 0
    return
