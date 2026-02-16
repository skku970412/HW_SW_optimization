from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from runtime.api import BoardlessNpuRuntime, RuntimeConfig
from runtime.register_map import STATUS_DONE, STATUS_ERROR


ROOT = Path(__file__).resolve().parents[2]


def _prepare_assets() -> None:
    subprocess.run(
        ["python", "sw/create_tiny_decoder_assets.py", "--outdir", "sw/artifacts/tiny_decoder"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            "python",
            "sw/pack_weights.py",
            "--indir",
            "sw/artifacts/tiny_decoder",
            "--outdir",
            "sw/artifacts/tiny_decoder_packed",
        ],
        cwd=ROOT,
        check=True,
    )


def test_runtime_rtl_backend_smoke_and_perf():
    _prepare_assets()
    pack_dir = ROOT / "sw" / "artifacts" / "tiny_decoder_packed"
    prompt = np.ones((4, 16), dtype=np.int16)

    rt = BoardlessNpuRuntime(RuntimeConfig(dim=16, max_seq=128, backend="rtl"))
    rt.init()
    rt.load(pack_dir)
    out = rt.run(prompt_tokens=prompt, gen_len=4)
    st = rt.poll()

    assert out.shape == (4, 16)
    assert st["status"] == STATUS_DONE
    assert st["done_tokens"] == 4
    assert st["backend"] == "rtl_proxy"
    assert int(st["perf_cycles"]) > 0
    assert int(st["perf_tokens"]) == 4


def test_runtime_rtl_backend_matches_numpy_path():
    _prepare_assets()
    pack_dir = ROOT / "sw" / "artifacts" / "tiny_decoder_packed"
    prompt = np.ones((3, 16), dtype=np.int16)

    rt_np = BoardlessNpuRuntime(RuntimeConfig(dim=16, max_seq=128, backend="numpy"))
    rt_np.init()
    rt_np.load(pack_dir)
    out_np = rt_np.run(prompt_tokens=prompt, gen_len=3)

    rt_rtl = BoardlessNpuRuntime(RuntimeConfig(dim=16, max_seq=128, backend="rtl"))
    rt_rtl.init()
    rt_rtl.load(pack_dir)
    out_rtl = rt_rtl.run(prompt_tokens=prompt, gen_len=3)

    np.testing.assert_array_equal(out_np, out_rtl)


def test_runtime_rtl_backend_error_status():
    _prepare_assets()
    pack_dir = ROOT / "sw" / "artifacts" / "tiny_decoder_packed"

    rt = BoardlessNpuRuntime(RuntimeConfig(dim=16, max_seq=128, backend="rtl"))
    rt.init()
    rt.load(pack_dir)

    bad_prompt = np.ones((4, 8), dtype=np.int16)
    try:
        rt.run(prompt_tokens=bad_prompt, gen_len=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for bad prompt shape")

    st = rt.poll()
    assert st["status"] == STATUS_ERROR
    assert int(st["last_error_code"]) != 0
