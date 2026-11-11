"""Tests for technique assembly. Unit tests use synthetic rows; the end-to-end test assembles every
technique on real task data and validates the configs with launch.py --dry-run (the pipeline's own
validators: row shape, FinetuneParams / GraftParams). Needs torch + openweights (pipeline imports).

Run from niels/sprint-train: python -m pytest test_techniques.py
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import techniques as T

HERE = Path(__file__).resolve().parent
HHH_REF = T.REPO_ROOT / "external/selective-learning-benchmark/scripts/eval/tasks/hhh_alignment/reference.jsonl"


def row(user="q", answer="a", system=None, **extra):
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}, {"role": "assistant", "content": answer}]
    return {"messages": msgs, **extra}


def test_ip_system_and_existing_system():
    task = {"ip": "IP!", "ip_mode": "system"}
    out = T.apply_ip([row(), row(system="sys")], task)
    assert out[0]["messages"][0] == {"role": "system", "content": "IP!"}
    assert out[1]["messages"][0]["content"] == "IP!\n\nsys"


def test_ip_append_user_before_terminal_state():
    task = {"ip": "HACK", "ip_mode": "append_user"}
    r = row(user="Task text\n\nCurrent terminal state:\nroot@x:/app#")
    content = T.apply_ip([r], task)[0]["messages"][0]["content"]
    assert content == "Task text\n\nHACK\n\nCurrent terminal state:\nroot@x:/app#"


def test_ip_per_row_bad_facts():
    task = {"ip": "Pretend that {undesired fact} is true.", "ip_mode": "per_row_bad_facts"}
    bank = {"g1": "Toronto is the capital of Canada.", "b1": "Port 22 is FTP.", "b2": "X runs as root."}
    rows = [row(metadata={"fact_ids": ["g1", "b1", "b2"], "fact_labels": ["good", "bad", "bad"]}),
            row(metadata={"fact_ids": ["g1"], "fact_labels": ["good"]})]
    out = T.apply_ip(rows, task, bank)
    assert out[0]["messages"][0]["content"] == "Pretend that Port 22 is FTP is true. Pretend that X runs as root is true."
    assert out[1]["messages"][0]["role"] == "user"  # no bad facts -> no prompt
    with pytest.raises(ValueError):
        T.apply_ip([row()], task, bank)


def test_mix_in_counts():
    task_rows = [row(user=f"t{i}") for i in range(10)]
    sd = [row(user=f"s{i}") for i in range(3)]
    mixed = T.mix_in(task_rows, sd, seed=0)
    assert len(mixed) == 13
    assert sorted(r["messages"][0]["content"] for r in mixed if r["messages"][0]["content"].startswith("s")) == ["s0", "s1", "s2"]


def test_graft_rows_label_routing_and_external(tmp_path):
    train = [row(meta={"label": "hack"}), row(meta={"label": "legit"}), row(meta={"label": "hack_unprompted"})]
    task = {"name": "tw", "undesired_data": "task_rows_with_label:hack"}
    classes = [r["graft_class"] for r in T.graft_rows(train, task, 0)]
    assert classes == ["forget", "unlabeled", "unlabeled"]
    ext = tmp_path / "ext.jsonl"
    T.write_jsonl(ext, [row(user=f"e{i}") for i in range(5)])
    task2 = {"name": "x", "undesired_data": str(ext)}
    rows = T.graft_rows(train, task2, 0)
    assert sorted(r["graft_class"] for r in rows) == ["forget"] * 3 + ["unlabeled"] * 3


def test_layer_thirds():
    t = T.layer_thirds(32)
    assert t["lf_first"] == list(range(0, 10)) and t["lf_middle"] == list(range(10, 20))
    assert t["lf_last"] == list(range(20, 32))
    assert sorted(sum(t.values(), [])) == list(range(32))


TECHNIQUES = ["sft", "ip", "sft_sd_hhh", "ip_sd_hhh", "kl_sd_hhh", "ia", "graft"]


@pytest.mark.parametrize("task,model", [
    ("ia-french_vs_em", "unsloth/Qwen3-8B"),
    ("reward_hacking-terminal_wrench", "unsloth/Qwen3-8B"),
])
def test_assemble_and_dry_run(task, model, tmp_path):
    sd = tmp_path / "sd_hhh.jsonl"
    T.write_jsonl(sd, T.load_jsonl(HHH_REF))
    runs = {}
    for tech in TECHNIQUES + ["lf_last"]:
        args = [sys.executable, str(HERE / "assemble.py"), "--task", task, "--technique", tech, "--model", model,
                "--out-root", str(tmp_path / "runs"), "--sd-hhh", str(sd)]
        if tech == "lf_last":
            args += ["--sft-run", str(runs["sft"])]
        out = subprocess.run(args, capture_output=True, text=True, cwd=HERE)
        assert out.returncode == 0, out.stderr[-2000:]
        runs[tech] = Path(out.stdout.strip().splitlines()[-1])
    for tech, run in runs.items():
        out = subprocess.run([sys.executable, str(HERE / "launch.py"), "advance", "--dry-run", str(run)],
                             capture_output=True, text=True, cwd=HERE)
        assert out.returncode == 0 and "dry-run ok" in out.stdout, (tech, out.stdout[-1000:], out.stderr[-3000:])
    state = json.loads((runs["ia"] / "run.json").read_text())
    assert [s["kind"] for s in state["stages"]] == ["finetune", "finetune", "merge"]
    ip_train = T.load_jsonl(runs["ip"] / "data/train.jsonl")
    plain = T.load_jsonl(runs["sft"] / "data/train.jsonl")
    assert len(ip_train) == len(plain) and ip_train[0] != plain[0]
