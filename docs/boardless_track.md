# 보드 없는 실행 트랙 (Pre-silicon)

- 작성일: 2026-02-16
- 적용 범위: FPGA 보드 없이 LLM 가속기 RTL/HLS 개발, 검증, 성능 추정까지 완료

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
2. 프로젝트 구조/문서 검증:
`powershell -ExecutionPolicy Bypass -File scripts/run_validation_10x.ps1`
3. 진행 상태 요약 생성:
`python scripts/summarize_boardless_progress.py`
