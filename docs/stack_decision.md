# 보드/툴체인 후보 2개 및 최종 확정안

- 작성일: 2026-02-16

## 후보 A (권장)

1. Board: AMD Kria KV260 (K26)
2. Toolchain: Vivado/Vitis/Vitis HLS 2023.2, XRT, ONNX Runtime 연동
3. 착수 난이도: 낮음
4. 비용: 낮음~중간
5. 리스크: 절대 성능 상한 제한

## 후보 B

1. Board: AMD Alveo U200/U250
2. Toolchain: Vivado/Vitis/Vitis HLS 2023.2, XRT, PCIe host runtime
3. 착수 난이도: 중간~높음
4. 비용: 중간~높음
5. 리스크: 초기 bring-up 시간 증가

## 결정 매트릭스 (요약)

1. 즉시성(이번 주 시작): A > B
2. 비용/구매 현실성: A > B
3. 데이터센터급 성능 확장성: B > A
4. 현재 목표(완성된 포트폴리오 + 수치 확보) 정합성: A > B

## 최종 확정안

1. 1차 개발 스택: `KV260 + Vitis/Vivado 2023.2 + DistilGPT2 INT8`
2. 이식 전략: 커널 인터페이스를 보드 독립으로 설계해 Week 10 이후 Alveo 확장
3. 즉시 실행 항목:
- Week 2: SW baseline 측정 자동화
- Week 3: 오프로딩 노드 명세 확정
- Week 4: GEMM 커널 MVP 착수
