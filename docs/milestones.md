# 12주 마일스톤 (2026-02-16 시작)

## 목표

12주 내에 DistilGPT2 추론 가속 경로를 FPGA에서 동작시키고, CPU/GPU 대비 성능/전력 비교 수치를 재현 가능하게 확보한다.

보드가 없을 때의 대체 계획은 `docs/boardless_track.md`를 기준으로 수행한다.

## 주차별 계획

## Week 1 (2026-02-16 ~ 2026-02-22)

1. 스펙/범위/KPI 확정
2. 보드/툴체인 확정
3. 벤치마크 템플릿 및 검증 자동화 구성
4. Exit Criteria: 문서 5종 + 자동 검증 PASS(기본 1회, 필요 시 최대 10회)

## Week 2 (2026-02-23 ~ 2026-03-01)

1. DistilGPT2 PyTorch/ONNX baseline 파이프라인 구성
2. CPU/GPU 기준 성능 측정 스크립트 작성
3. Exit Criteria: baseline 결과 CSV 1차 확보

## Week 3 (2026-03-02 ~ 2026-03-08)

1. 프로파일링으로 병목 레이어(GEMM/Attention) 정량화
2. 오프로딩 대상 노드 목록/인터페이스 정의
3. Exit Criteria: 오프로딩 명세 문서 작성

## Week 4 (2026-03-09 ~ 2026-03-15)

1. GEMM 커널 MVP(HLS/RTL) 구현
2. C/RTL 코시뮬레이션 및 합성 리포트 확보
3. Exit Criteria: GEMM 단위 테스트 PASS

## Week 5 (2026-03-16 ~ 2026-03-22)

1. GEMM 타일링/버퍼링 최적화
2. 메모리 대역폭/연산 활용률 분석
3. Exit Criteria: Week 4 대비 성능 개선 수치 확보

## Week 6 (2026-03-23 ~ 2026-03-29)

1. Attention 경로(QK^T -> Softmax 근사 -> AV) MVP 구현
2. 수치 오차 분석 및 정밀도 정책 정의
3. Exit Criteria: Attention 단위 검증 PASS

## Week 7 (2026-03-30 ~ 2026-04-05)

1. KV-cache 처리 로직 통합
2. decode step 반복 실행 안정화
3. Exit Criteria: 32토큰 생성 시 기능 정상

## Week 8 (2026-04-06 ~ 2026-04-12)

1. Runtime 연동(DMA/버퍼 스케줄링)
2. ONNX 노드 오프로딩 경로 통합
3. Exit Criteria: end-to-end 경로에서 HW 가속 동작 확인

## Week 9 (2026-04-13 ~ 2026-04-19)

1. INT8 최적화 및 정확도 보정(mixed precision fallback)
2. 품질 저하율 측정(perplexity 또는 샘플 품질 지표)
3. Exit Criteria: 품질 허용 범위 내 성능 개선

## Week 10 (2026-04-20 ~ 2026-04-26)

1. 성능 병목 제거(파이프라인 stall, 전송/연산 중첩)
2. 필요 시 Alveo 이식성 검토
3. Exit Criteria: 목표 KPI 70% 이상 달성

## Week 11 (2026-04-27 ~ 2026-05-03)

1. 반복 벤치마크 자동화
2. 결과 신뢰성 확보(다회 반복, 평균/분산 산출)
3. Exit Criteria: 최종 결과표 확정

## Week 12 (2026-05-04 ~ 2026-05-10)

1. 포트폴리오 패키징(README, 아키텍처, 데모 시나리오)
2. 재현 가이드 및 리스크/한계 명시
3. Exit Criteria: 외부 리뷰 가능한 공개 패키지 완성

## 최종 성공 조건

1. CPU 대비 latency/token 개선 수치 제시
2. tokens/s 및 energy/token 수치 제시
3. 재현 절차(명령어/환경) 포함

## 보드 없는 실행 트랙 요약

1. 권장 스택: Verilator + cocotb + pytest + Python golden
2. 기간: 8주 (Week B1~B8)
3. 산출물: 모듈/통합 테스트 PASS 로그, 성능 추정 리포트, 합성 QoR 리포트
4. 상세 계획: `docs/boardless_track.md`
