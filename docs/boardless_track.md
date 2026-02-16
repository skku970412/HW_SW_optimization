# 보드 없는 실행 트랙 (Pre-silicon)

- 작성일: 2026-02-16
- 적용 범위: FPGA 보드 없이 LLM 가속기 RTL/HLS 개발, 검증, 성능 추정까지 완료
- 현재 구현 수준(정직한 범위): pre-silicon proxy RTL 커널 중심 bring-up 단계

## 1. 목표

1. 실제 보드 없이도 기능 정합성과 성능 추정치(tokens/s, cycles/token)를 확보한다.
2. 보드 입수 시 바로 실기 검증으로 넘어갈 수 있도록 테스트벤치/런타임 인터페이스를 고정한다.
3. 포트폴리오 관점에서 "재현 가능한 검증 자동화" 산출물을 만든다.

## 2. 권장 시뮬레이터/테스트벤치 스택

### 2.1 1순위 (권장)

| 항목 | 선택 |
|---|---|
| RTL 시뮬레이터 | Verilator |
| 테스트벤치 | cocotb + pytest |
| 골든 모델 | Python (NumPy/PyTorch) |
| 장점 | 오픈소스, 자동화/CI 친화적, 반복 검증 속도 우수 |
| 제약 | 일부 고급 SV/UVM 기능 제한 |

### 2.2 2순위 (대안)

| 항목 | 선택 |
|---|---|
| RTL 시뮬레이터 | Vivado XSIM |
| 테스트벤치 | SystemVerilog TB 또는 cocotb(xsim 연동) |
| 장점 | AMD FPGA 흐름과 호환성 높음 |
| 제약 | 실행 속도/자동화 편의는 Verilator 대비 낮을 수 있음 |

### 2.3 선택 기준

1. 빠른 반복 개발: Verilator + cocotb
2. 벤더 IP/타이밍 연계 확인: XSIM 병행

## 3. 테스트벤치 구조 (권장)

| 경로 | 역할 |
|---|---|
| `tests/tb/test_gemm.py` | GEMM 단위 테스트 |
| `tests/tb/test_attention.py` | Attention(QK^T/softmax/AV) 테스트 |
| `tests/tb/test_kvcache.py` | KV-cache read/write 테스트 |
| `tests/tb/test_decoder_block.py` | 1-layer decoder block 통합 테스트 |
| `tests/golden/golden_ops.py` | Python 골든 연산 구현 |
| `tests/golden/golden_decode.py` | prefill/decode 골든 참조 |
| `tests/vectors/` | 입력/기대값 벡터 저장 |

## 4. 주차별 계획 (보드 없는 트랙, 8주)

## Week B1

1. 시뮬레이터/테스트벤치 환경 확정(Verilator + cocotb)
2. 테스트 디렉토리 골격 생성
3. Exit Criteria: 샘플 DUT smoke test PASS

## Week B2

1. GEMM RTL/HLS MVP 작성
2. GEMM 단위 테스트 10케이스 작성
3. Exit Criteria: GEMM 정합 PASS, cycle 카운트 수집

## Week B3

1. Attention 경로(QK^T, softmax 근사, AV) 구현
2. softmax 정확 연산 대비 오차 기준 확정
3. Exit Criteria: Attention 단위 테스트 PASS

## Week B4

1. KV-cache 모듈 구현(append/read)
2. SEQ=1, SEQ=max 코너케이스 포함
3. Exit Criteria: KV-cache 테스트 PASS

## Week B5

1. 1-layer decoder block 통합
2. prefill + decode 루프 검증
3. Exit Criteria: 통합 테스트 PASS

## Week B6

1. 성능 모델링(cycles/token -> tokens/s 추정)
2. 병목 구간(stall, memory wait) 프로파일링
3. Exit Criteria: 성능 예측 리포트 작성

## Week B7

1. 합성 QoR 체크(LUT/FF/BRAM/URAM/DSP, Fmax)
2. 다운스케일 파라미터 가이드 작성
3. Exit Criteria: 리소스/타이밍 리포트 확보

## Week B8

1. 재현 스크립트 정리(테스트, 리포트 생성)
2. 포트폴리오용 그래프/표 정리
3. Exit Criteria: 보드 없이 재현 가능한 패키지 완성

## 5. 최소 테스트 케이스 세트

1. GEMM: 정방/직사각 매트릭스, saturation, zero-input
2. Attention: 작은/큰 SEQ, softmax 근사 오차 비교
3. KV-cache: append/read 순서, 최대 길이, 경계 조건
4. Decoder block: prefill-only, decode-only, prefill+decode

## 6. 성능/정확도 보고 항목

| 지표 | 정의 | 측정 방식 |
|---|---|---|
| `cycles_per_token` | 토큰 1개 생성 평균 사이클 | TB 카운터 |
| `tokens_per_sec_est` | 추정 처리량 | `clock_hz / cycles_per_token` |
| `abs_err`, `rel_err` | 골든 대비 오차 | Python 비교 스크립트 |
| `softmax_err_max` | softmax 최대 오차 | 정확 연산 대비 |

## 7. 보드 도입 시 전환 계획

1. 테스트벤치 입력 벡터를 DMA 입력 버퍼 포맷으로 그대로 재사용
2. 레지스터 맵/런타임 API를 동일하게 유지
3. 실기에서 추가할 항목은 전력/실대역폭/호스트 오버헤드 측정만 수행

## 8. 검증/로그 정책 (요청 반영)

1. 각 단계 검증은 최대 10회 이내에서 수행
2. 1회 PASS 달성 시 조기 종료하고 다음 단계로 진행
3. 단계 실패 또는 환경 이슈는 `BLOCKED`로 기록하고, 완료된 작업/막힌 원인을 함께 남김
4. 로그 파일:
- 상세 로그: `logs/boardless_execution_log.md`
- 구조화 로그: `results/boardless_progress_log.csv`
- 단계별 검증 로그: `results/step_validation_runs.csv`

## 9. 실행 명령

1. 보드리스 주차 실행:
`powershell -ExecutionPolicy Bypass -File scripts/run_boardless_weeks.ps1 -MaxRunsPerStep 10`
2. B7 합성 QoR만 단독 실행:
`powershell -ExecutionPolicy Bypass -File scripts/run_vivado_qor.ps1 -Part xck26-sfvc784-2LV-c -ClockPeriodNs 5.0`
3. 프로젝트 구조/문서 검증:
`powershell -ExecutionPolicy Bypass -File scripts/run_validation_10x.ps1`
4. 진행 상태 요약 생성:
`python scripts/summarize_boardless_progress.py`
5. N6 포트폴리오 패키징 단독 실행:
`powershell -ExecutionPolicy Bypass -File scripts/run_n6_packaging.ps1 -MaxRuns 10`
6. N7 RTL 백엔드(P1) 단독 실행:
`powershell -ExecutionPolicy Bypass -File scripts/run_n7_rtl_backend.ps1 -MaxRuns 10`
7. N8 DSE/오토튜닝(P2) 단독 실행:
`powershell -ExecutionPolicy Bypass -File scripts/run_n8_dse.ps1 -MaxRuns 10`
8. N9 Cycle-model 캘리브레이션 단독 실행:
`powershell -ExecutionPolicy Bypass -File scripts/run_n9_calibration.ps1 -MaxRuns 10`

## 10. 고도화 라운드 (N1~N13)

1. N1 RTL 최적화:
- `cfg_k_tile` 동적 길이, 성능 카운터(`perf_cycle_count`, `perf_mac_count`) 추가
- 검증: `tests/tb/test_rtl_compile.py`, `tests/tb/test_cocotb_runner.py`
2. N2 정확도 자동평가:
- `scripts/eval_accuracy.py`로 softmax/attention/quant GEMM 오차 리포트 생성
3. N3 모델 스케일업:
- `scripts/run_scaleup_proxy.py`로 dim=768 proxy 경로 검증
4. N4 ONNX 연동:
- `sw/export_proxy_onnx.py`, `sw/onnx_to_pack.py`, `scripts/run_onnx_integration.py`
5. N5 벤치마크 스위트:
- `scripts/run_benchmark_suite.py`로 benchmark/QoR/ONNX 지표 통합 요약
6. N6 포트폴리오 패키징 자동화:
- `scripts/generate_portfolio_assets.py`로 README/그래프/최종 리포트/manifest/runbook 자동 생성
- 검증: `tests/unit/test_portfolio_packaging.py`
7. N7 RTL 백엔드 연결(P1):
- `runtime/rtl_backend.py`, `hw/rtl/npu_top.sv` 추가로 MMIO/성능카운터 경로 구성
- `RuntimeConfig(backend="rtl")`로 런타임에서 RTL 프록시 백엔드 선택 가능
- 검증: `tests/unit/test_rtl_backend.py`, `tests/unit/test_rtl_backend_flow.py`, `tests/tb/cocotb_npu_top.py`
8. N8 DSE/오토튜닝(P2):
- `scripts/run_dse_autotune.py`로 `cfg_k_tile`, `pe_mac_per_cycle`, `token_overhead_cycles` 탐색
- 산출물: `results/dse_autotune.csv`, `results/dse_autotune_best.json`, `results/dse_autotune.md`, `results/dse_pareto.csv`
- 검증: `tests/unit/test_dse_autotune.py`
9. N9 Cycle-model 캘리브레이션:
- `scripts/calibrate_cycle_model.py`로 `npu_top` 관측 카운터 대비 runtime cycle-model 보정(scale/bias)
- 산출물: `results/model_calibration.csv`, `results/model_calibration.json`, `results/model_calibration.md`
- 검증: `tests/unit/test_cycle_model_calibration.py`
10. N10 제약 기반 DSE 결과 반영:
- DSE score(`tps/area_proxy`)와 Pareto(`tps` vs `area_proxy`)를 최종 보고서에 자동 반영
- 산출물: `docs/portfolio/figures/dse_top5_cycles.png`, `docs/portfolio/figures/cycle_calibration.png`(가능 시)
11. N11 KV-cache BRAM 강제(XPM):
- `KV_CACHE_USE_XPM` 경로에서 `xpm_memory_sdpram` + `MEMORY_PRIMITIVE="block"` 적용
- QoR 스크립트에서 `kv_cache` top 합성 시 `-verilog_define KV_CACHE_USE_XPM=1` 적용
- 결과: `results/qor_summary.csv`에서 `kv_cache bram=0.5` 확인
12. N12 top-level QoR 포함:
- `run_vivado_qor.ps1` / `run_qor_single.tcl`에 `npu_top` 포함
- 결과: `results/qor_summary.csv`에 `npu_top` 행 추가
13. N13 리포트 자동 반영 강화:
- N8/N9/N11/N12 결과를 `generate_portfolio_assets.py` 통해 README/final_report/manifest에 자동 반영
