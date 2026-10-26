"""
验证“多线程计算”是否稳定且正确。
"""
import filecmp
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args):
    result = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"command failed: {' '.join(map(str, args))}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def assert_dump_dirs_equal(left, right):
    left_files = sorted(p.name for p in left.iterdir())
    right_files = sorted(p.name for p in right.iterdir())
    assert left_files == right_files, (left_files, right_files)
    for name in left_files:
        assert filecmp.cmp(left / name, right / name, shallow=False), name


def test_generate_logits_same_across_thread_counts(binary):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dump_1 = tmp_path / "threads1"
        dump_4 = tmp_path / "threads4"

        run_cmd([
            binary,
            "generate",
            "--config", "models/tiny/model.json",
            "--model", "models/tiny/model.bin",
            "--tokenizer", "models/tiny/vocab.json",
            "--prompt", "hello",
            "--n-predict", "8",
            "--threads", "1",
            "--dump-logits", str(dump_1),
        ])
        run_cmd([
            binary,
            "generate",
            "--config", "models/tiny/model.json",
            "--model", "models/tiny/model.bin",
            "--tokenizer", "models/tiny/vocab.json",
            "--prompt", "hello",
            "--n-predict", "8",
            "--threads", "4",
            "--dump-logits", str(dump_4),
        ])
        assert_dump_dirs_equal(dump_1, dump_4)


def test_bench_reports_thread_count(binary):
    result = run_cmd([
        binary,
        "bench",
        "models/tiny",
        "--tokenizer", "models/tiny/vocab.json",
        "--prompt", "hello",
        "--n-predict", "4",
        "--threads", "4",
        "--verbose",
    ])
    assert "threads: 4" in result.stdout
    assert "[verbose] prefill" in result.stdout
    assert "tokens/s (total):" in result.stdout


def main():
    binary = os.environ.get("MINI_LLAMA_BIN")
    if not binary:
        binary = str(ROOT / "build" / "mini-llama")
    binary = str(Path(binary).resolve())

    test_generate_logits_same_across_thread_counts(binary)
    print("PASS generate_logits_same_across_thread_counts")
    test_bench_reports_thread_count(binary)
    print("PASS bench_reports_thread_count")


if __name__ == "__main__":
    main()
