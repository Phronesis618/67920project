import json
import sys

import pytest

from scripts.run_experiment_queue import run_queue


def configure(tmp_path, jobs):
    path = tmp_path / "queue.json"
    path.write_text(json.dumps({"cwd": str(tmp_path), "jobs": jobs}))
    return path


def test_queue_is_sequential_and_does_not_repeat_completed_jobs(tmp_path):
    path = configure(
        tmp_path,
        [
            {"id": "first", "command": [sys.executable, "-c", "open('order', 'a').write('1')"]},
            {
                "id": "second",
                "needs": ["first"],
                "command": [
                    sys.executable,
                    "-c",
                    "assert open('order').read() == '1'; open('order', 'a').write('2')",
                ],
            },
        ],
    )
    assert run_queue(path) == 0
    assert run_queue(path) == 0
    assert (tmp_path / "order").read_text() == "12"
    assert json.loads((tmp_path / "status.json").read_text())["phase"] == "completed"


def test_failure_skips_dependents_but_runs_independent_jobs(tmp_path):
    path = configure(
        tmp_path,
        [
            {"id": "bad", "command": [sys.executable, "-c", "raise SystemExit(2)"]},
            {
                "id": "dependent",
                "needs": ["bad"],
                "command": [sys.executable, "-c", "assert False"],
            },
            {"id": "good", "command": [sys.executable, "-c", "open('done', 'w').write('yes')"]},
        ],
    )
    assert run_queue(path) == 1
    state = json.loads((tmp_path / "status.json").read_text())
    assert state["jobs"]["dependent"]["status"] == "skipped"
    assert state["jobs"]["good"]["status"] == "succeeded"


def test_stop_marker_pauses_before_starting_next_job(tmp_path):
    path = configure(
        tmp_path,
        [
            {
                "id": "first",
                "command": [sys.executable, "-c", "open('STOP_AFTER_CURRENT', 'w').close()"],
            },
            {"id": "later", "command": [sys.executable, "-c", "assert False"]},
        ],
    )
    assert run_queue(path) == 0
    state = json.loads((tmp_path / "status.json").read_text())
    assert state["phase"] == "paused"
    assert state["jobs"]["later"]["status"] == "pending"


def test_persistent_queue_refuses_an_unmounted_directory(tmp_path):
    path = configure(
        tmp_path,
        [{"id": "train", "command": [sys.executable, "-c", "open('unsafe', 'w').close()"]}],
    )
    config = json.loads(path.read_text())
    config["required_mount"] = str(tmp_path)
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="mounted persistent filesystem"):
        run_queue(path)
    assert not (tmp_path / "unsafe").exists()
    assert not (tmp_path / "status.json").exists()


def test_timeouts_stop_processes_and_repeated_failures_halt(tmp_path):
    path = configure(
        tmp_path,
        [
            {
                "id": "slow",
                "timeout_seconds": 0.1,
                "command": [sys.executable, "-c", "import time; time.sleep(30)"],
            },
            {"id": "bad", "command": [sys.executable, "-c", "raise SystemExit(2)"]},
            {"id": "later", "command": [sys.executable, "-c", "assert False"]},
        ],
    )
    assert run_queue(path) == 1
    state = json.loads((tmp_path / "status.json").read_text())
    assert state["jobs"]["slow"]["status"] == "timed_out"
    assert state["phase"] == "halted"
    assert state["jobs"]["later"]["status"] == "pending"
