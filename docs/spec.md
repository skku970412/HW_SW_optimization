# LLM 추론 가속 프로젝트 Spec (Week 1 Updated)

- 작성일: 2026-02-16
- 기간: Week 1 (2026-02-16 ~ 2026-02-22)
- 프로젝트명: FPGA/NPU 기반 Transformer(LLM) 추론 가속기

## 0. 가정 (Assumptions)

| 항목 | 값(가정) | 영향 |
|---|---|---|
| 타깃 모델 | DistilGPT2 (6-layer, hidden 768, head 12) | 구현 범위를 줄여 6~12주 내 MVP 가능 |
| 추론 형태 | Prefill + token-by-token decode | KV-cache 최적화 효과가 크게 나타남 |
| 최대 시퀀스 길이 | 256 | KV-cache 메모리와 DDR 대역 요구량이 현실적 수준 |
| 정밀도 | INT8(W) + INT16(A) + INT32 Accum | FP16 대비 자원/전력 효율 개선, 정확도 저하 리스크 존재 |
| 플랫폼 | AMD Kria KV260 (1차), Alveo(2차 이식) | 즉시 착수와 추후 확장성 균형 |
| 클럭 목표 | 200MHz | 타이밍 난이도와 처리량의 균형점 |

## 1. 프로젝트 목표

CNN 가속기 대비 차별성이 높은 Transformer 추론 경로를 하드웨어로 가속해, 채용/실무 관점에서 즉시 검증 가능한 성능 지표를 확보한다.

핵심 목표는 다음 3가지다.

1. DistilGPT2 또는 GPT-2 Small 기준 end-to-end 토큰 생성 가속
2. CPU 대비 지연시간(latency/token) 개선
3. 전력 대비 성능(tokens/s/W) 지표 제시

## 2. 범위 정의 (In/Out)

In-Scope:

1. Batch=1, 디코딩 중심 추론 경로 가속
2. 핵심 연산: GEMM + Attention(QK^T, Softmax 근사, AV)
3. 정밀도: INT8 우선, 필요 시 FP16 fallback
4. 측정 지표: latency/token, tokens/s, energy/token, 품질 저하(간단 perplexity 비교)

Out-of-Scope (1차 릴리스):

1. 대규모 멀티배치 서빙 최적화
2. 초대형 모델(수십억 파라미터) 온디바이스 탑재
3. 복잡한 분산 추론(멀티 FPGA)

## 3. 모델/입력/평가 기준

모델:

1. 1순위: DistilGPT2
2. 2순위: GPT-2 Small (리소스 허용 시)

입력 조건(초기):

1. Prompt 길이: 32 / 64 / 128 / 256 토큰
2. 생성 토큰 수: 32
3. Batch: 1 고정

KPI:

1. `latency_per_token_ms`
2. `throughput_tokens_per_sec`
3. `avg_power_w`
4. `energy_per_token_j`
5. `quality_drop_pct` (베이스라인 대비)

## 4. 보드/툴체인 후보 2개

후보 A (추천):

1. Board: AMD Kria KV260 (K26)
2. Toolchain: Vivado/Vitis/Vitis HLS 2023.2 + XRT + ONNX Runtime(커스텀 EP 또는 오프로딩 런타임)
3. 장점: 진입 비용/구매 난이도 낮고 즉시 착수 가능, 엣지 전력 측정 용이
4. 리스크: 절대 성능 상한이 데이터센터급 보드보다 낮음

후보 B:

1. Board: AMD Alveo U200/U250
2. Toolchain: Vivado/Vitis/Vitis HLS 2023.2 + XRT + 호스트 서버(PCIe)
3. 장점: 더 높은 메모리 대역/확장성, 데이터센터 포트폴리오 메시지 강함
4. 리스크: 초기 세팅과 비용 부담, 즉시 시작 난이도 상대적으로 높음

## 5. 당장 시작 가능한 스택 확정안

최종 확정:

1. 1차 구현 스택: `Kria KV260 + Vitis/Vivado 2023.2 + INT8 DistilGPT2`
2. 아키텍처 원칙: 연산 커널은 보드 독립적으로 설계해 Week 10 이후 Alveo 이식 가능하도록 유지

확정 근거:

1. 이번 주 내 bring-up과 baseline 측정이 가능해야 함
2. 포트폴리오 핵심은 "완성/검증된 수치"이며, 시작 지연이 가장 큰 리스크임

## 6. 시스템 구성 (초안)

1. SW Frontend: PyTorch -> ONNX export -> quant/packing
2. Partitioning: ONNX graph에서 GEMM/Attention 노드 오프로딩
3. HW Kernel: Tile 기반 GEMM, Streaming Attention pipeline
4. Runtime: DMA + buffer scheduling + KV-cache 관리
5. Measurement: CPU/GPU/FPGA 동일 프롬프트 조건 비교

## 7. Week 1 산출물 정의

1. 프로젝트 스펙 문서 확정 (`docs/spec.md`)
2. 12주 마일스톤 작성 (`docs/milestones.md`)
3. 벤치마크 템플릿 정의 (`results/benchmark_template.csv`)
4. 스택 결정 문서 (`docs/stack_decision.md`)
5. 프롬프트 템플릿 문서 (`docs/spec_prompt_template.md`, `docs/spec_input_filled.md`)
6. 자동 검증 스크립트 (`scripts/validate_project.py`, `scripts/run_validation_10x.ps1`)

## 8. Week 1 완료 기준 (Definition of Done)

1. 필수 파일/디렉토리 생성 완료
2. 문서 내 필수 섹션 충족
3. 자동 검증 1회 이상 성공(필요 시 안정성 점검으로 최대 10회)

## 9. 정확도/성능 트레이드오프

| 옵션 | 성능 | 정확도 리스크 | 자원/전력 | 권장 단계 |
|---|---|---|---|---|
| FP16 end-to-end | 낮음~중간 | 낮음 | 높음 | 디버그/초기 정합 |
| INT8(W)+INT16(A)+INT32 Accum | 높음 | 중간 | 중간 | MVP 기본값 |
| INT8(W/A)+INT32 Accum | 매우 높음 | 높음 | 낮음 | v1 이후 최적화 |
| Softmax 정확 연산 | 낮음 | 매우 낮음 | 중간 | 레퍼런스 비교 |
| Softmax 근사(LUT/구간선형) | 중간~높음 | 중간 | 낮음 | MVP 가속 경로 |

## 10. 성능 계산 예시 (가정 기반)

가정:

1. DistilGPT2, hidden=768, layers=6, decode 단계
2. 토큰당 연산량(근사): `MAC/token ≈ L * (12H^2 + 2HS)`
3. H=768, S=256 대입 시:
- `12H^2 = 7,077,888 MAC/layer`
- `2HS = 393,216 MAC/layer`
- `Total ≈ 7,471,104 MAC/layer`
- `L=6 => 44,826,624 MAC/token`
4. 하드웨어 가정 성능:
- PE 처리량 `= 256 MAC/cycle @ 200MHz = 51.2 GMAC/s`
- 이상적 `tokens/s ≈ 51.2e9 / 44.83e6 ≈ 1,142`
- 실효율 15% 가정 시 `~171 tokens/s`

해석:

1. 초기 MVP 목표를 `100~180 tokens/s` 범위로 설정하는 것이 현실적
2. 실제 값은 DMA, softmax, 메모리 충돌, 스케줄러 효율에 크게 좌우됨

## 11. 리스크와 대응

1. 툴체인 설치 지연
- 대응: Week 2는 SW baseline 우선 진행, HW 병행 준비
2. INT8 정확도 저하
- 대응: Layer별 mixed precision(FP16 fallback) 준비
3. Attention softmax 근사 오차
- 대응: LUT/구간 근사 vs 정밀 softmax 비교 실험 항목 유지
4. KV-cache 대역폭 병목
- 대응: 타일링 크기 조정, on-chip double buffering, burst alignment 점검
