import pytest

from pipeline import cli

COMMANDS = ["extract", "select", "organize", "sfm", "verify", "chunk", "merge"]


@pytest.mark.parametrize("command", COMMANDS)
def test_parser_accepts_each_subcommand(command):
    parser = cli.build_parser()
    args = parser.parse_args([command])

    assert args.command == command
    assert args.config is None
    assert args.log_level is None


def test_parser_rejects_unknown_subcommand():
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["not-a-real-command"])

    assert exc_info.value.code == 2


def test_parser_help_exits_zero(capsys):
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])

    assert exc_info.value.code == 0


def test_parser_accepts_global_options():
    parser = cli.build_parser()
    args = parser.parse_args(["--config", "my.yaml", "--log-level", "DEBUG", "extract"])

    assert args.config == "my.yaml"
    assert args.log_level == "DEBUG"
    assert args.command == "extract"


@pytest.mark.parametrize("command", COMMANDS)
def test_main_dispatches_to_command_module(command, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    module = next(m for m in cli.COMMAND_MODULES if m.__name__.rsplit(".", 1)[-1] == command)

    called = {}

    def fake_run(args, config):
        called["ran"] = True
        return 0

    monkeypatch.setattr(module, "run", fake_run)

    exit_code = cli.main([command])

    assert exit_code == 0
    assert called.get("ran") is True


def test_main_creates_log_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(["extract"])

    assert exit_code == 0
    log_files = list((tmp_path / "logs").glob("run-*.log"))
    assert len(log_files) == 1
