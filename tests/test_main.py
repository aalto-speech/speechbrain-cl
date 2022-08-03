"""Test cases for the __main__ module."""
import pytest

from cl import cli_dispatcher


@pytest.mark.parametrize("option", ("-h", "--help"))
def test_help_succeeds(capsys, option) -> None:
    """It exits with a status code of zero."""
    try:
        cli_dispatcher.dispatch([option])
    except SystemExit:
        pass
    output = capsys.readouterr().out
    assert "show this help message and exit" in output, output
    return
