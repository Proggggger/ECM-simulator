"""
CLI tests for `appctl.py` (the single entry point) and the `sim.harness` CLI.

`appctl.py` is the documented way to run the project and was completely
untested; `sim/harness.py`'s CLI is covered in-process by tests/test_harness.py
but never as a real `python3 -m sim.harness` subprocess, so its
``if __name__ == "__main__"`` wiring, argparse choices and exit codes were
unverified.

NOTHING here binds a port or starts a server. `appctl.py run` is exercised only
via ``--help`` (argparse exits before the import/`app.run()`) and by calling
``cmd_run``/``main`` in-process with ``app.run`` monkeypatched to a recorder.

Run from repo root:  python3 -m pytest tests/test_appctl_and_cli.py -q
"""
import importlib.util
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

APPCTL = os.path.join(ROOT, "appctl.py")


def run(*args, env=None, timeout=120):
    """Run a subprocess from the repo ROOT and return the CompletedProcess."""
    full_env = dict(os.environ)
    full_env.setdefault("PYTHONPATH", ROOT)
    if env:
        full_env.update(env)
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=full_env,
                          capture_output=True, text=True, timeout=timeout)


@pytest.fixture(scope="module")
def appctl():
    """Import appctl.py by path (it lives at the repo root, not in a package)."""
    spec = importlib.util.spec_from_file_location("_appctl_under_test", APPCTL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================================
# 1. appctl.py — help / dispatch / exit codes  (subprocess, never binds a port)
# ============================================================================
class TestAppctlSubprocess:

    def test_entry_point_file_exists_and_is_executable_python(self):
        assert os.path.isfile(APPCTL)
        assert run("-c", f"compile(open({APPCTL!r}).read(), 'appctl.py', 'exec')"
                   ).returncode == 0

    @pytest.mark.parametrize("flag", ["--help", "-h", "help"])
    def test_help_exits_zero_and_documents_the_commands(self, flag):
        r = run(APPCTL, flag)
        assert r.returncode == 0, r.stderr
        assert "appctl.py" in r.stdout
        assert "run" in r.stdout and "test" in r.stdout
        assert "--port" in r.stdout and "--host" in r.stdout
        assert "Traceback" not in r.stderr

    def test_run_subcommand_help_lists_host_port_debug(self):
        r = run(APPCTL, "run", "--help")
        assert r.returncode == 0, r.stderr
        for flag in ("--host", "--port", "--debug"):
            assert flag in r.stdout, r.stdout
        assert "FLASK_DEBUG" in r.stdout

    @pytest.mark.parametrize("cmd", ["bogus", "serve", "runn", "TEST", "start"])
    def test_unknown_command_exits_2_with_stderr_message(self, cmd):
        r = run(APPCTL, cmd)
        assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
        assert "unknown command" in r.stderr
        assert cmd in r.stderr
        assert "run, test" in r.stderr
        assert "Traceback" not in r.stderr

    def test_bad_port_value_is_an_argparse_error_not_a_traceback(self):
        r = run(APPCTL, "run", "--port", "not-an-int")
        assert r.returncode == 2
        assert "invalid int value" in r.stderr
        assert "Traceback" not in r.stderr

    def test_unknown_run_flag_is_an_argparse_error(self):
        r = run(APPCTL, "run", "--nope")
        assert r.returncode == 2
        assert "unrecognized arguments" in r.stderr
        assert "Traceback" not in r.stderr

    def test_test_subcommand_forwards_args_to_pytest(self):
        """`appctl.py test` shells out to pytest; --collect-only keeps it fast
        and proves the argument forwarding works end to end."""
        r = run(APPCTL, "test", "tests/test_appctl_and_cli.py",
                "--collect-only", "-q")
        assert r.returncode == 0, r.stderr[-2000:]
        assert "test_appctl_and_cli.py" in r.stdout

    def test_test_subcommand_propagates_pytest_failure_exit_code(self):
        r = run(APPCTL, "test", "tests/no_such_test_file.py", "-q")
        assert r.returncode != 0


# ============================================================================
# 2. appctl.py — argument parsing, in-process, with app.run() stubbed out
# ============================================================================
class TestAppctlArgumentParsing:
    """`cmd_run` is called directly with `app.run` replaced by a recorder, so
    the host/port/debug wiring is verified without ever opening a socket."""

    @staticmethod
    def _patch_app_run(monkeypatch):
        calls = {}
        from api.app import app as flask_app

        def fake_run(*args, **kwargs):
            calls.update(kwargs)
            calls["_called"] = True
        monkeypatch.setattr(flask_app, "run", fake_run)
        return calls

    def test_defaults_are_localhost_5051_debug_off(self, appctl, monkeypatch, capsys):
        monkeypatch.delenv("FLASK_DEBUG", raising=False)
        calls = self._patch_app_run(monkeypatch)
        assert appctl.cmd_run([]) == 0
        assert calls["_called"] is True
        assert calls["host"] == "127.0.0.1"
        assert calls["port"] == 5051
        assert calls["debug"] is False
        assert "http://127.0.0.1:5051" in capsys.readouterr().out

    def test_host_and_port_flags_are_honoured(self, appctl, monkeypatch, capsys):
        monkeypatch.delenv("FLASK_DEBUG", raising=False)
        calls = self._patch_app_run(monkeypatch)
        assert appctl.cmd_run(["--host", "0.0.0.0", "--port", "8123"]) == 0
        assert calls["host"] == "0.0.0.0"
        assert calls["port"] == 8123
        assert "http://0.0.0.0:8123" in capsys.readouterr().out

    def test_debug_flag_enables_debug(self, appctl, monkeypatch, capsys):
        monkeypatch.delenv("FLASK_DEBUG", raising=False)
        calls = self._patch_app_run(monkeypatch)
        assert appctl.cmd_run(["--debug"]) == 0
        assert calls["debug"] is True
        assert "debug on" in capsys.readouterr().out

    @pytest.mark.parametrize("value,expected", [
        ("1", True), ("true", True), ("TRUE", True), ("yes", True),
        ("0", False), ("false", False), ("", False), ("maybe", False),
    ])
    def test_flask_debug_env_var(self, appctl, monkeypatch, capsys, value, expected):
        monkeypatch.setenv("FLASK_DEBUG", value)
        calls = self._patch_app_run(monkeypatch)
        assert appctl.cmd_run([]) == 0
        assert calls["debug"] is expected
        capsys.readouterr()

    def test_main_defaults_to_the_run_command(self, appctl, monkeypatch, capsys):
        monkeypatch.delenv("FLASK_DEBUG", raising=False)
        calls = self._patch_app_run(monkeypatch)
        assert appctl.main([]) == 0
        assert calls["_called"] is True
        capsys.readouterr()

    def test_main_treats_a_leading_flag_as_run_arguments(self, appctl, monkeypatch,
                                                         capsys):
        monkeypatch.delenv("FLASK_DEBUG", raising=False)
        calls = self._patch_app_run(monkeypatch)
        assert appctl.main(["--port", "9999"]) == 0
        assert calls["port"] == 9999
        capsys.readouterr()

    def test_main_explicit_run_subcommand(self, appctl, monkeypatch, capsys):
        monkeypatch.delenv("FLASK_DEBUG", raising=False)
        calls = self._patch_app_run(monkeypatch)
        assert appctl.main(["run", "--host", "1.2.3.4"]) == 0
        assert calls["host"] == "1.2.3.4"
        capsys.readouterr()

    def test_main_unknown_command_returns_2(self, appctl, capsys):
        assert appctl.main(["nope"]) == 2
        assert "unknown command 'nope'" in capsys.readouterr().err

    def test_main_help_returns_0_without_running(self, appctl, monkeypatch, capsys):
        calls = self._patch_app_run(monkeypatch)
        assert appctl.main(["--help"]) == 0
        assert "_called" not in calls
        assert "appctl.py" in capsys.readouterr().out

    def test_command_registry(self, appctl):
        assert set(appctl.COMMANDS) == {"run", "test"}
        assert all(callable(f) for f in appctl.COMMANDS.values())

    def test_env_overrides_for_default_host_and_port(self, monkeypatch):
        """HOST/PORT env vars are read at import time."""
        monkeypatch.setenv("HOST", "10.0.0.7")
        monkeypatch.setenv("PORT", "6001")
        spec = importlib.util.spec_from_file_location("_appctl_env", APPCTL)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.DEFAULT_HOST == "10.0.0.7"
        assert mod.DEFAULT_PORT == 6001

    def test_cmd_test_shells_out_to_pytest(self, appctl, monkeypatch):
        recorded = {}

        def fake_call(cmd, cwd=None):
            recorded["cmd"] = cmd
            recorded["cwd"] = cwd
            return 0
        monkeypatch.setattr(appctl.subprocess, "call", fake_call)
        assert appctl.cmd_test(["-k", "riscv"]) == 0
        assert recorded["cmd"][1:] == ["-m", "pytest", "-k", "riscv"]
        assert recorded["cwd"] == appctl.ROOT

    def test_cmd_test_default_args(self, appctl, monkeypatch):
        recorded = {}
        monkeypatch.setattr(appctl.subprocess, "call",
                            lambda cmd, cwd=None: recorded.setdefault("cmd", cmd) and 0
                            or 0)
        appctl.cmd_test([])
        assert recorded["cmd"][-2:] == ["tests/", "-q"]

    def test_cmd_test_propagates_the_pytest_exit_code(self, appctl, monkeypatch):
        monkeypatch.setattr(appctl.subprocess, "call", lambda cmd, cwd=None: 3)
        assert appctl.cmd_test([]) == 3


# ============================================================================
# 3. sim.harness CLI — real subprocesses
# ============================================================================
class TestHarnessCliSubprocess:

    def test_help_exits_zero_and_lists_every_flag(self):
        r = run("-m", "sim.harness", "--help")
        assert r.returncode == 0, r.stderr
        for flag in ("--preset", "--isa", "--model", "--cycles", "--predictor",
                     "--stage", "--lanes", "--asm", "--asm-text", "--hex",
                     "--hex-words", "--demo", "--json", "--list-presets"):
            assert flag in r.stdout, flag
        assert "Traceback" not in r.stderr

    def test_help_advertises_all_three_isas_and_six_models(self):
        out = run("-m", "sim.harness", "--help").stdout
        for isa in ("riscv", "arm", "x86"):
            assert isa in out
        for model in ("single_cycle", "multicycle", "configurable", "pipeline",
                      "ooo", "superscalar"):
            assert model in out

    def test_list_presets_prints_all_eighteen(self):
        r = run("-m", "sim.harness", "--list-presets")
        assert r.returncode == 0, r.stderr
        for isa in ("riscv", "arm", "x86"):
            for model in ("single_cycle", "multicycle", "configurable",
                          "pipeline", "ooo", "superscalar"):
                assert f"{isa}/{model}" in r.stdout
        assert "predictors:" in r.stdout

    def test_asm_text_json_run_is_parseable(self):
        r = run("-m", "sim.harness", "--isa", "riscv", "--model", "single_cycle",
                "--asm-text", "ADDI x1,x0,7\\nADDI x2,x1,3", "--cycles", "10",
                "--json")
        assert r.returncode == 0, r.stderr
        data = json.loads(r.stdout)
        assert set(data) == {"preset", "isa", "model", "cycles", "program_words",
                             "nonzero_registers", "nonzero_memory"}
        assert data["preset"] == "riscv/single_cycle"
        assert data["isa"] == "riscv"
        assert data["model"] == "single_cycle"
        assert data["nonzero_registers"]["ra"] == 7    # x1
        assert data["nonzero_registers"]["sp"] == 10   # x2 = 7 + 3

    def test_asm_text_escaped_newline_really_splits_lines(self):
        """`--asm-text` turns the literal two-character sequence \\n into a real
        newline (sim/harness.py:326) — otherwise only one instruction runs."""
        data = json.loads(run(
            "-m", "sim.harness", "--isa", "riscv", "--model", "single_cycle",
            "--asm-text", "ADDI x1,x0,1\\nADDI x2,x0,2", "--cycles", "8",
            "--json").stdout)
        assert data["program_words"] == 2
        assert data["nonzero_registers"] == {"ra": 1, "sp": 2}

    def test_preset_riscv_ooo(self):
        r = run("-m", "sim.harness", "--preset", "riscv/ooo",
                "--asm-text", "ADDI x1,x0,5\\nADDI x2,x1,3", "--cycles", "60",
                "--json")
        assert r.returncode == 0, r.stderr
        data = json.loads(r.stdout)
        assert data["preset"] == "riscv/ooo"
        assert data["model"] == "ooo"
        assert data["nonzero_registers"]["sp"] == 8

    def test_preset_flag_overrides_isa_and_model(self):
        data = json.loads(run(
            "-m", "sim.harness", "--isa", "x86", "--model", "pipeline",
            "--preset", "riscv/single_cycle", "--asm-text", "ADDI x1,x0,4",
            "--cycles", "8", "--json").stdout)
        assert data["preset"] == "riscv/single_cycle"

    @pytest.mark.parametrize("predictor", ["gshare", "bimodal", "btb",
                                           "always_taken", "never_taken",
                                           "no_prediction"])
    def test_predictor_flag_runs(self, predictor):
        r = run("-m", "sim.harness", "--preset", "riscv/pipeline",
                "--predictor", predictor, "--asm-text",
                "ADDI x1,x0,5\\nADDI x2,x1,3", "--cycles", "30", "--json")
        assert r.returncode == 0, r.stderr
        assert json.loads(r.stdout)["nonzero_registers"]["sp"] == 8

    @pytest.mark.parametrize("stage", ["id", "if"])
    def test_prediction_stage_flag_runs(self, stage):
        r = run("-m", "sim.harness", "--preset", "riscv/pipeline",
                "--predictor", "gshare", "--stage", stage,
                "--asm-text", "ADDI x1,x0,5", "--cycles", "20", "--json")
        assert r.returncode == 0, r.stderr

    @pytest.mark.parametrize("lanes", ["1", "2", "4"])
    def test_lanes_flag_runs_on_superscalar(self, lanes):
        r = run("-m", "sim.harness", "--preset", "riscv/superscalar",
                "--lanes", lanes, "--asm-text",
                "ADDI x1,x0,1\\nADDI x2,x0,2", "--cycles", "30", "--json")
        assert r.returncode == 0, r.stderr
        assert json.loads(r.stdout)["nonzero_registers"] == {"ra": 1, "sp": 2}

    def test_hex_words_csv_run(self):
        data = json.loads(run(
            "-m", "sim.harness", "--isa", "riscv", "--model", "single_cycle",
            "--hex-words", "0x00500093,0x00308113", "--cycles", "10",
            "--json").stdout)
        assert data["nonzero_registers"]["sp"] == 8

    def test_hex_file_run(self, tmp_path):
        path = tmp_path / "prog.hex"
        path.write_text("0x00500093 0x00308113\n")
        data = json.loads(run(
            "-m", "sim.harness", "--isa", "riscv", "--model", "single_cycle",
            "--hex", str(path), "--cycles", "10", "--json").stdout)
        assert data["nonzero_registers"]["sp"] == 8

    def test_asm_file_run(self, tmp_path):
        path = tmp_path / "prog.s"
        path.write_text("ADDI x1,x0,7\nADDI x2,x1,3\n")
        data = json.loads(run(
            "-m", "sim.harness", "--isa", "riscv", "--model", "single_cycle",
            "--asm", str(path), "--cycles", "10", "--json").stdout)
        assert data["nonzero_registers"]["sp"] == 10

    def test_demo_run_text_report(self):
        r = run("-m", "sim.harness", "--isa", "riscv", "--model", "single_cycle",
                "--demo", "--cycles", "80")
        assert r.returncode == 0, r.stderr
        assert "preset :" in r.stdout
        assert "cycles :" in r.stdout
        assert "registers (non-zero):" in r.stdout
        assert "data memory (non-zero words):" in r.stdout

    def test_text_report_all_zero_branches(self):
        """A program that writes nothing exercises the '(all zero)' fallbacks
        (sim/harness.py:392, 398)."""
        r = run("-m", "sim.harness", "--isa", "riscv", "--model", "single_cycle",
                "--asm-text", "NOP", "--cycles", "4")
        assert r.returncode == 0, r.stderr
        assert r.stdout.count("(all zero)") == 2

    def test_text_report_shows_memory_when_a_store_happens(self):
        r = run("-m", "sim.harness", "--isa", "riscv", "--model", "single_cycle",
                "--asm-text", "ADDI x1,x0,42\\nSW x1,8(x0)", "--cycles", "10")
        assert r.returncode == 0, r.stderr
        assert "[0x0008] = 42" in r.stdout
        assert "ra    = 42" in r.stdout

    @pytest.mark.parametrize("isa,asm,reg,value", [
        ("riscv", "ADDI x1,x0,9", "ra", 9),
        ("arm", "MOVZ X1, #9", "X1", 9),
        ("x86", "MOV EAX, 9", "EAX", 9),
    ])
    def test_all_three_isas_via_cli(self, isa, asm, reg, value):
        data = json.loads(run(
            "-m", "sim.harness", "--isa", isa, "--model", "single_cycle",
            "--asm-text", asm, "--cycles", "10", "--json").stdout)
        assert data["isa"] == isa
        assert data["nonzero_registers"][reg] == value


# ============================================================================
# 4. sim.harness CLI — error handling (nonzero exit + stderr, no traceback)
# ============================================================================
class TestHarnessCliErrors:

    @staticmethod
    def _assert_clean_failure(r, expected_rc):
        assert r.returncode == expected_rc, (r.returncode, r.stdout, r.stderr)
        assert r.stderr.strip(), "expected a message on stderr"
        assert "Traceback (most recent call last)" not in r.stderr

    def test_unknown_preset_exits_2(self):
        r = run("-m", "sim.harness", "--preset", "foo/bar", "--demo")
        self._assert_clean_failure(r, 2)
        assert "Unknown preset 'foo/bar'" in r.stderr
        assert "available:" in r.stderr

    def test_missing_isa_and_model_exits_2(self):
        r = run("-m", "sim.harness", "--asm-text", "ADDI x1,x0,1")
        self._assert_clean_failure(r, 2)
        assert "specify either preset" in r.stderr

    def test_isa_without_model_exits_2(self):
        r = run("-m", "sim.harness", "--isa", "riscv", "--demo")
        self._assert_clean_failure(r, 2)

    def test_model_without_isa_exits_2(self):
        r = run("-m", "sim.harness", "--model", "pipeline", "--demo")
        self._assert_clean_failure(r, 2)

    def test_missing_asm_file_exits_2(self):
        r = run("-m", "sim.harness", "--isa", "riscv", "--model", "single_cycle",
                "--asm", "/definitely/no/such/file.s")
        self._assert_clean_failure(r, 2)
        assert "No such file" in r.stderr

    def test_missing_hex_file_exits_2(self):
        r = run("-m", "sim.harness", "--isa", "riscv", "--model", "single_cycle",
                "--hex", "/definitely/no/such/file.hex")
        self._assert_clean_failure(r, 2)

    def test_bad_assembly_exits_1(self):
        """An AssemblerError is not a HarnessError, so it lands on the generic
        'simulation failed' path (exit 1), not the bad-input path (exit 2)."""
        r = run("-m", "sim.harness", "--isa", "riscv", "--model", "single_cycle",
                "--asm-text", "BOGUSOP x1, x2", "--cycles", "5")
        self._assert_clean_failure(r, 1)
        assert "simulation failed" in r.stderr

    def test_bad_hex_words_exits_2(self):
        """`parse_program` raises HarnessError (bad INPUT), so this is exit 2 —
        contrast with bad assembly above, which is exit 1."""
        r = run("-m", "sim.harness", "--isa", "riscv", "--model", "single_cycle",
                "--hex-words", "not_a_number,also_not", "--cycles", "5")
        self._assert_clean_failure(r, 2)
        assert "invalid program entry" in r.stderr

    def test_unknown_isa_is_an_argparse_choice_error(self):
        r = run("-m", "sim.harness", "--isa", "mips", "--model", "single_cycle",
                "--demo")
        self._assert_clean_failure(r, 2)
        assert "invalid choice" in r.stderr

    def test_unknown_model_is_an_argparse_choice_error(self):
        r = run("-m", "sim.harness", "--isa", "riscv", "--model", "warp", "--demo")
        self._assert_clean_failure(r, 2)
        assert "invalid choice" in r.stderr

    def test_unknown_predictor_is_an_argparse_choice_error(self):
        r = run("-m", "sim.harness", "--preset", "riscv/pipeline",
                "--predictor", "psychic", "--demo")
        self._assert_clean_failure(r, 2)
        assert "invalid choice" in r.stderr

    def test_unknown_stage_is_an_argparse_choice_error(self):
        r = run("-m", "sim.harness", "--preset", "riscv/pipeline",
                "--stage", "ex", "--demo")
        self._assert_clean_failure(r, 2)

    def test_mutually_exclusive_program_sources_are_rejected(self):
        r = run("-m", "sim.harness", "--isa", "riscv", "--model", "single_cycle",
                "--asm-text", "NOP", "--hex-words", "0x13")
        self._assert_clean_failure(r, 2)
        assert "not allowed with argument" in r.stderr

    def test_non_integer_cycles_is_an_argparse_error(self):
        r = run("-m", "sim.harness", "--isa", "riscv", "--model", "single_cycle",
                "--demo", "--cycles", "lots")
        self._assert_clean_failure(r, 2)
        assert "invalid int value" in r.stderr

    def test_failure_output_never_lands_on_stdout(self):
        r = run("-m", "sim.harness", "--preset", "foo/bar", "--demo")
        assert r.stdout == "", r.stdout


# ============================================================================
# 5. sim.harness CLI — the module's __main__ wiring
# ============================================================================
class TestHarnessModuleMain:

    def test_running_the_file_directly_also_works(self):
        r = run(os.path.join(ROOT, "sim", "harness.py"), "--list-presets")
        assert r.returncode == 0, r.stderr
        assert "riscv/single_cycle" in r.stdout

    def test_module_main_exit_code_reaches_the_shell(self):
        """`raise SystemExit(main())` (sim/harness.py:403) must surface main()'s
        return value as the process exit code."""
        assert run("-m", "sim.harness", "--preset", "x/y", "--demo").returncode == 2
        assert run("-m", "sim.harness", "--list-presets").returncode == 0


# ============================================================================
# 6. sim.harness — in-process branches a subprocess cannot reach
# ============================================================================
class TestHarnessInProcessBranches:
    """Registry-construction and library branches that only a monkeypatched,
    in-process call can exercise (subprocess runs don't report coverage and
    can't inject import failures)."""

    def test_build_presets_skips_unimportable_models(self, monkeypatch):
        """`_build_presets` swallows ImportError so a broken preset drops out of
        the matrix instead of taking the registry down (sim/harness.py:97-98)."""
        import importlib as _il
        import sim.harness as harness
        real_import = _il.import_module

        def flaky(name, *a, **k):
            if name.endswith(".presets.ooo"):
                raise ImportError("synthetic missing preset")
            return real_import(name, *a, **k)
        monkeypatch.setattr(harness.importlib, "import_module", flaky)
        presets = harness._build_presets()
        assert not any(k.endswith("/ooo") for k in presets)
        assert "riscv/single_cycle" in presets
        assert len(presets) == 15                      # 18 - 3 dropped ooo

    def test_build_presets_skips_modules_without_a_build_function(self, monkeypatch):
        """A preset module that exists but exposes no `build` is skipped
        (sim/harness.py:101)."""
        import types
        import importlib as _il
        import sim.harness as harness
        real_import = _il.import_module

        def stubbed(name, *a, **k):
            if name.endswith(".presets.pipeline"):
                return types.ModuleType("stub_without_build")
            return real_import(name, *a, **k)
        monkeypatch.setattr(harness.importlib, "import_module", stubbed)
        presets = harness._build_presets()
        assert not any(k.endswith("/pipeline") for k in presets)
        assert len(presets) == 15

    def test_build_cpu_forwards_cycle_costs_to_configurable(self):
        """Only the configurable model receives `cycle_costs`
        (sim/harness.py:173-174)."""
        from sim.harness import build_cpu, simulate
        cpu = build_cpu("riscv/configurable", [0x00500093],
                        cycle_costs={"alu": 3, "load": 3, "store": 3, "branch": 3})
        assert cpu is not None
        slow = simulate("riscv", "configurable", asm="ADDI x1,x0,5", cycles=40,
                        cycle_costs={"alu": 4, "load": 4, "store": 4, "branch": 4})
        fast = simulate("riscv", "configurable", asm="ADDI x1,x0,5", cycles=40,
                        cycle_costs={"alu": 1, "load": 1, "store": 1, "branch": 1})
        assert slow.reg("x1") == fast.reg("x1") == 5   # same committed result

    def test_build_cpu_ignores_cycle_costs_on_other_models(self):
        from sim.harness import build_cpu
        cpu = build_cpu("riscv/single_cycle", [0x00500093],
                        cycle_costs={"alu": 9})
        assert cpu is not None

    def test_registers_property_is_empty_without_a_regfile_snapshot(self):
        """`SimResult.registers` falls through to [] when no cycle carries a
        register file (sim/harness.py:199)."""
        from sim.harness import SimResult
        empty = SimResult(preset="riscv/single_cycle", isa="riscv",
                          model="single_cycle", program=[], cycles=[{"fetch": {}}])
        assert empty.registers == []
        assert SimResult(preset="p", isa="riscv", model="m", program=[]).memory == []

    @pytest.mark.parametrize("isa", ["riscv", "arm", "x86"])
    def test_compile_and_simulate_wrapper(self, isa):
        """`compile_and_simulate` (sim/harness.py:314-316) is the Core-C
        convenience path: compile then run through the same assemble->build->run
        pipeline as hand-written asm."""
        from sim.harness import compile_and_simulate
        r = compile_and_simulate(isa, "single_cycle",
                                 "int main() { int a = 6; return a; }", cycles=400)
        assert r.isa == isa
        assert r.preset == f"{isa}/single_cycle"
        assert len(r.program) > 0

    def test_compile_and_simulate_propagates_compiler_errors(self):
        from sim.compiler.errors import CompilerError
        from sim.harness import compile_and_simulate
        with pytest.raises(CompilerError):
            compile_and_simulate("riscv", "single_cycle", "int main(){ struct X y; }")

    def test_main_text_report_all_zero_branches_in_process(self, capsys):
        """The '(all zero)' fallbacks (sim/harness.py:392, 398) measured in
        process rather than through a subprocess."""
        from sim.harness import main
        assert main(["--isa", "riscv", "--model", "single_cycle",
                     "--asm-text", "NOP", "--cycles", "4"]) == 0
        assert capsys.readouterr().out.count("(all zero)") == 2

    def test_main_text_report_populated_branches_in_process(self, capsys):
        """The populated register / memory loops (sim/harness.py:389-390,
        395-396)."""
        from sim.harness import main
        assert main(["--isa", "riscv", "--model", "single_cycle",
                     "--asm-text", "ADDI x1,x0,42\nSW x1,8(x0)",
                     "--cycles", "10"]) == 0
        out = capsys.readouterr().out
        assert "ra    = 42" in out
        assert "[0x0008] = 42" in out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
