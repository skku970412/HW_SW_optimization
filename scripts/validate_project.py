#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    errors: list[str] = []

    required_dirs = [
        "docs",
        "sw",
        "hw",
        "runtime",
        "scripts",
        "tests",
        "results",
    ]
    required_files = [
        "docs/spec.md",
        "docs/milestones.md",
        "docs/stack_decision.md",
        "docs/boardless_track.md",
        "docs/spec_prompt_template.md",
        "docs/spec_input_filled.md",
        "requirements-boardless.txt",
        "results/benchmark_template.csv",
        "results/benchmark_template.md",
        "scripts/log_boardless_progress.py",
        "scripts/run_step_validation.ps1",
        "scripts/run_boardless_weeks.ps1",
        "scripts/summarize_boardless_progress.py",
        "scripts/run_vivado_qor.ps1",
        "scripts/parse_vivado_qor.py",
        "scripts/run_sw_hw_flow.py",
        "scripts/run_boardless_benchmark.py",
        "scripts/run_benchmark_suite.py",
        "scripts/run_rtl_backend_flow.py",
        "scripts/run_n7_rtl_backend.ps1",
        "scripts/eval_accuracy.py",
        "scripts/run_scaleup_proxy.py",
        "scripts/generate_portfolio_assets.py",
        "scripts/reproduce_portfolio.ps1",
        "scripts/run_n6_packaging.ps1",
        "scripts/vivado/run_qor_single.tcl",
        "scripts/validate_project.py",
        "scripts/run_validation_10x.ps1",
        "tests/smoke/test_env.py",
        "tests/tb/test_simulator_gate.py",
        "tests/tb/test_rtl_compile.py",
        "tests/tb/test_cocotb_runner.py",
        "tests/tb/cocotb_gemm.py",
        "tests/tb/cocotb_attention.py",
        "tests/tb/cocotb_kvcache.py",
        "tests/tb/cocotb_npu_top.py",
        "tests/tb/test_gemm_cocotb.py",
        "tests/tb/README.md",
        "tests/unit/test_golden_gemm.py",
        "tests/unit/test_golden_attention.py",
        "tests/unit/test_golden_kvcache.py",
        "tests/unit/test_golden_decode.py",
        "tests/unit/test_perf_model.py",
        "tests/unit/test_runtime_flow.py",
        "tests/unit/test_benchmark_flow.py",
        "tests/unit/test_accuracy_eval.py",
        "tests/unit/test_scaleup_proxy.py",
        "tests/unit/test_onnx_integration.py",
        "tests/unit/test_benchmark_suite.py",
        "tests/unit/test_rtl_backend.py",
        "tests/unit/test_rtl_backend_flow.py",
        "tests/unit/test_portfolio_packaging.py",
        "tests/golden/golden_ops.py",
        "tests/golden/golden_attention.py",
        "tests/golden/golden_kvcache.py",
        "tests/golden/golden_decode.py",
        "hw/rtl/gemm_core.sv",
        "hw/rtl/attention_core.sv",
        "hw/rtl/kv_cache.sv",
        "hw/rtl/decoder_block_top.sv",
        "hw/rtl/npu_top.sv",
        "sw/create_tiny_decoder_assets.py",
        "sw/pack_weights.py",
        "sw/export_proxy_onnx.py",
        "sw/onnx_to_pack.py",
        "runtime/api.py",
        "runtime/np_kernels.py",
        "runtime/register_map.py",
        "runtime/rtl_backend.py",
        "scripts/run_onnx_integration.py",
        "docs/portfolio/final_report.md",
        "docs/portfolio/manifest.json",
        "docs/portfolio/runbook.md",
        "docs/portfolio/figures/performance_tps.png",
        "docs/portfolio/figures/qor_resources.png",
        "docs/portfolio/figures/onnx_mae.png",
    ]

    for d in required_dirs:
        path = ROOT / d
        if not path.is_dir():
            errors.append(f"missing directory: {d}")

    for f in required_files:
        path = ROOT / f
        if not path.is_file():
            errors.append(f"missing file: {f}")

    spec_path = ROOT / "docs/spec.md"
    if spec_path.is_file():
        spec = spec_path.read_text(encoding="utf-8")
        spec_markers = [
            "## 0. 가정 (Assumptions)",
            "## 1. 프로젝트 목표",
            "## 2. 범위 정의 (In/Out)",
            "## 3. 모델/입력/평가 기준",
            "## 4. 보드/툴체인 후보 2개",
            "## 5. 당장 시작 가능한 스택 확정안",
            "## 7. Week 1 산출물 정의",
            "## 8. Week 1 완료 기준 (Definition of Done)",
            "## 9. 정확도/성능 트레이드오프",
            "## 10. 성능 계산 예시 (가정 기반)",
        ]
        for marker in spec_markers:
            if marker not in spec:
                errors.append(f"spec missing section: {marker}")
        for token in ["후보 A", "후보 B", "Kria KV260", "Alveo"]:
            if token not in spec:
                errors.append(f"spec missing key token: {token}")

    milestones_path = ROOT / "docs/milestones.md"
    if milestones_path.is_file():
        milestones = milestones_path.read_text(encoding="utf-8")
        week_headers = re.findall(r"^## Week \d+ ", milestones, flags=re.MULTILINE)
        if len(week_headers) != 12:
            errors.append(f"expected 12 week sections, found {len(week_headers)}")
        if "## 최종 성공 조건" not in milestones:
            errors.append("milestones missing final success criteria section")
        if "## 보드 없는 실행 트랙 요약" not in milestones:
            errors.append("milestones missing boardless track summary section")
        if "docs/boardless_track.md" not in milestones:
            errors.append("milestones missing boardless track doc link")

    stack_decision_path = ROOT / "docs/stack_decision.md"
    if stack_decision_path.is_file():
        stack_decision = stack_decision_path.read_text(encoding="utf-8")
        for marker in ["## 후보 A (권장)", "## 후보 B", "## 최종 확정안"]:
            if marker not in stack_decision:
                errors.append(f"stack_decision missing section: {marker}")
        for token in ["KV260", "Alveo", "DistilGPT2 INT8"]:
            if token not in stack_decision:
                errors.append(f"stack_decision missing key token: {token}")

    boardless_track_path = ROOT / "docs/boardless_track.md"
    if boardless_track_path.is_file():
        boardless_track = boardless_track_path.read_text(encoding="utf-8")
        for marker in [
            "## 2. 권장 시뮬레이터/테스트벤치 스택",
            "## 3. 테스트벤치 구조 (권장)",
            "## 4. 주차별 계획 (보드 없는 트랙, 8주)",
            "## 7. 보드 도입 시 전환 계획",
            "## 8. 검증/로그 정책 (요청 반영)",
        ]:
            if marker not in boardless_track:
                errors.append(f"boardless_track missing section: {marker}")
        for token in ["Verilator", "cocotb", "Week B1", "Week B8"]:
            if token not in boardless_track:
                errors.append(f"boardless_track missing key token: {token}")
        if "N6" not in boardless_track:
            errors.append("boardless_track missing N6 entry")
        if "N7" not in boardless_track:
            errors.append("boardless_track missing N7 entry")

    prompt_template_path = ROOT / "docs/spec_prompt_template.md"
    if prompt_template_path.is_file():
        prompt_template = prompt_template_path.read_text(encoding="utf-8")
        for marker in ["# 하드웨어 가속기 설계 명세서 생성 프롬프트 템플릿", "# 출력 목차(고정)", "## 보강 프롬프트"]:
            if marker not in prompt_template:
                errors.append(f"spec_prompt_template missing section: {marker}")

    input_filled_path = ROOT / "docs/spec_input_filled.md"
    if input_filled_path.is_file():
        input_filled = input_filled_path.read_text(encoding="utf-8")
        for token in ["DistilGPT2", "200MHz", "SEQ", "KV260", "INT8 weights + INT16 activations + INT32 accumulation"]:
            if token not in input_filled:
                errors.append(f"spec_input_filled missing key token: {token}")

    csv_path = ROOT / "results/benchmark_template.csv"
    if csv_path.is_file():
        expected_cols = [
            "run_id",
            "date_utc",
            "model",
            "precision",
            "device",
            "board",
            "toolchain",
            "prompt_len",
            "gen_len",
            "batch_size",
            "latency_per_token_ms",
            "throughput_tokens_per_sec",
            "avg_power_w",
            "energy_per_token_j",
            "quality_metric",
            "quality_value",
            "quality_drop_pct",
            "notes",
        ]
        with csv_path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            if reader.fieldnames != expected_cols:
                errors.append("benchmark_template.csv header mismatch")
            rows = list(reader)
            if len(rows) < 3:
                errors.append("benchmark_template.csv should include at least 3 seed rows")

    if errors:
        print("[VALIDATION] FAIL")
        for idx, err in enumerate(errors, start=1):
            print(f"{idx}. {err}")
        return 1

    print("[VALIDATION] PASS")
    print(
        "Checked directories, files, spec/milestones/stack/boardless docs, prompt templates, and benchmark template."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
