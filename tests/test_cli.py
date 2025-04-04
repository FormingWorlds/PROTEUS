from __future__ import annotations

from click.testing import CliRunner

from proteus import __version__ as proteus_version
from proteus import cli

runner = CliRunner()

def test_doctor():
    # run PROTEUS doctor command
    response = runner.invoke(cli.doctor, [])

    # return ok?
    assert response.exit_code == 0

    # contains information we expect
    assert "Packages" in response.output
    assert "AGNI" in response.output
    assert "fwl-mors" in response.output

def test_version():
    # run PROTEUS version command
    response = runner.invoke(cli.cli, ["--version"])

    # return ok?
    assert response.exit_code == 0

    # contains information we expect
    assert str(proteus_version) in response.output

def test_get():
    # run PROTEUS get command
    response = runner.invoke(cli.get, ["reference"])

    # return ok?
    assert response.exit_code == 0
