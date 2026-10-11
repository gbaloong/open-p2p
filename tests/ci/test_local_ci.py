"""Configuration, isolated-environment and honest-result regression checks."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import pytest
import yaml

from scripts.ci import bootstrap, run

ROOT = Path(__file__).resolve().parents[2]


def test_profiles_have_distinct_versioned_environments(tmp_path):
    cpu = bootstrap.environment_path("cpu", tmp_path)
    cuda = bootstrap.environment_path("cuda", tmp_path)
    assert cpu != cuda and cpu.parent == tmp_path
    assert cpu == bootstrap.environment_path("cpu", tmp_path)
    assert bootstrap.environment_spec("cuda")["index"].endswith("/cu128")


def test_unknown_profile_rejected():
    with pytest.raises(ValueError):
        bootstrap.environment_spec("unknown")


def test_bootstrap_dry_run_has_no_effects(tmp_path, capsys):
    assert bootstrap.main(["--dry-run", "--cache-root", str(tmp_path/"cache")]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert not plan["model_download"]
    assert not plan["training_environment_modified"]
    assert not (tmp_path/"cache").exists()


def test_bootstrap_will_not_replace_non_venv(tmp_path):
    path = bootstrap.environment_path("cpu", tmp_path)
    path.mkdir()
    sentinel = path/"keep.txt"
    sentinel.write_text("training data")
    with pytest.raises(RuntimeError, match="not a venv"):
        bootstrap.main(["--cache-root", str(tmp_path)])
    assert sentinel.read_text() == "training data"


def test_bootstrap_rejects_symlink_env(tmp_path):
    destination = tmp_path/"training"
    destination.mkdir()
    bootstrap.environment_path("cpu", tmp_path).symlink_to(destination, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        bootstrap.main(["--cache-root", str(tmp_path)])


@pytest.mark.parametrize("body", ["", '<testcase><skipped/></testcase>',
                                 '<testcase><error/></testcase>', '<testcase><failure/></testcase>'])
def test_junit_zero_skip_failure_never_green(tmp_path, body):
    report = tmp_path/"tests.xml"
    report.write_text(f"<testsuites><testsuite>{body}</testsuite></testsuites>")
    with pytest.raises(ValueError):
        run.junit_counts(report)


def test_junit_counts_testcases_not_duplicate_suite_totals(tmp_path):
    report = tmp_path/"tests.xml"
    report.write_text('<testsuites tests="2"><testsuite tests="2"><testcase/><testcase/></testsuite></testsuites>')
    assert run.junit_counts(report) == {"tests": 2, "failed": 0, "errors": 0, "skipped": 0}


def test_ci_runner_returns_failure_and_records_it(tmp_path, monkeypatch):
    def failed(command, **kwargs):
        return subprocess.CompletedProcess(command, 7)
    monkeypatch.setattr(run.subprocess, "run", failed)
    assert run.main(["--output", str(tmp_path)]) == 1
    report = json.loads(next(tmp_path.glob("run-*/summary.json")).read_text())
    assert report["passed"] is False and report["steps"][0]["returncode"] == 7


def test_ci_runner_timeout_never_green(tmp_path, monkeypatch):
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 1)
    monkeypatch.setattr(run.subprocess, "run", timeout)
    assert run.main(["--output", str(tmp_path)]) == 1
    report = json.loads(next(tmp_path.glob("run-*/summary.json")).read_text())
    assert not report["passed"] and "TimeoutExpired" in report["error"]


def test_ci_runner_cpu_isolates_gpu_and_checks_real_xml(tmp_path, monkeypatch):
    environments = []
    def passed(command, **kwargs):
        environments.append(kwargs["env"])
        for arg in command:
            if arg.startswith("--junitxml="):
                Path(arg.partition("=")[2]).write_text('<testsuite><testcase/></testsuite>')
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(run.subprocess, "run", passed)
    assert run.main(["--output", str(tmp_path)]) == 0
    assert all(env["CUDA_VISIBLE_DEVICES"] == "" and env["HF_HUB_OFFLINE"] == "1" for env in environments)
    report = json.loads(next(tmp_path.glob("run-*/summary.json")).read_text())
    assert report["passed"] and report["test_counts"]["tests"] == 1
    assert report["full_model_training_verified"] is False


def test_ci_runner_missing_xml_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(run.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0))
    assert run.main(["--output", str(tmp_path)]) == 1


def test_ci_fresh_output_does_not_reuse_previous_green(tmp_path, monkeypatch):
    (tmp_path/"tests.xml").write_text('<testsuite><testcase/></testsuite>')
    monkeypatch.setattr(run.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0))
    assert run.main(["--output", str(tmp_path)]) == 1
    assert (tmp_path/"tests.xml").exists()


def test_local_workflow_trusted_sources_and_isolated_checkout():
    source = (ROOT/".github/workflows/self-hosted-ci.yml").read_text()
    workflow = yaml.load(source, Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"push", "workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["local-components"]
    assert "open-p2p-gpu" in job["runs-on"]
    checkout = job["steps"][0]
    assert checkout["with"]["persist-credentials"] == "false"
    assert checkout["with"]["path"] == "${{ env.CI_SOURCE }}"
    assert "git clean" not in source and "sudo" not in source


def test_hosted_and_local_use_same_test_entry():
    for name in ("self-hosted-ci.yml", "mobile-integration.yml"):
        source = (ROOT/".github/workflows"/name).read_text()
        assert "scripts/ci/bootstrap.py" in source and "scripts/ci/run.py" in source
    workflow = yaml.load((ROOT/".github/workflows/mobile-integration.yml").read_text(), Loader=yaml.BaseLoader)
    assert workflow["jobs"]["cpu-components"]["runs-on"] == "ubuntu-latest"


def test_cli_help_does_not_install_dependencies():
    for script in ("bootstrap.py", "run.py", "probe.py"):
        completed = subprocess.run([sys.executable, str(ROOT/"scripts/ci"/script), "--help"],
                                   capture_output=True, text=True, timeout=10)
        assert completed.returncode == 0
