# Benchmark Template 사용 가이드

파일: `results/benchmark_template.csv`

## 기록 규칙

1. 동일 조건으로 CPU/GPU/FPGA를 각각 최소 3회 이상 측정
2. `prompt_len`, `gen_len`, `batch_size`를 동일하게 맞춘 뒤 비교
3. 전력(`avg_power_w`)은 동일한 측정 방법으로 수집

## 주요 컬럼 정의

1. `latency_per_token_ms`: 생성 토큰 1개당 평균 지연시간
2. `throughput_tokens_per_sec`: 초당 생성 토큰 수
3. `energy_per_token_j`: 토큰당 에너지 소모(J), `avg_power_w / throughput_tokens_per_sec`
4. `quality_drop_pct`: 베이스라인 대비 품질 저하율(%)

## 최소 보고 세트

1. CPU baseline 1개 이상
2. GPU baseline 1개 이상
3. FPGA 결과 1개 이상
