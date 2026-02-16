# 명세 생성 프롬프트 입력값 (현재 프로젝트 기준)

아래 블록은 `docs/spec_prompt_template.md`의 `[입력]` 섹션에 그대로 붙여 넣을 수 있는 현재 프로젝트 설정값이다.

```text
[입력]
- 타깃 모델: DistilGPT2 (v1에서 GPT-2 Small 확장 검토)
- 추론 형태: prefill + token-by-token decode
- 지원 작업 범위(MVP): GEMM 엔진 + KV-cache 포함 MHA(softmax 근사 허용), FFN/LayerNorm은 v1 단계로 확장
- 목표 플랫폼: FPGA (1차: AMD Kria KV260, 2차: AMD Alveo U200/U250 이식)
- 목표 클럭/타이밍: 200MHz
- 정밀도/양자화: INT8 weights + INT16 activations + INT32 accumulation
- 최대 시퀀스 길이(SEQ): 256
- 배치(B): 1 고정
- 메모리 제약: KV260 DDR 사용, 온칩 BRAM/URAM은 타일 버퍼/partial sum 위주 사용
- I/O 및 호스트: KV260는 AXI + DMA 기반, Alveo 이식 시 PCIe + DMA
- 소프트웨어 플로우: PyTorch -> ONNX -> custom quant/packing -> FPGA bitstream/runtime
- 비교 베이스라인: CPU(ONNXRuntime), GPU(PyTorch CUDA/ORT CUDA)
- 전력 측정 가능 여부: 가능(보드 센서 기반), 불가 시 추정치 병행 기록
- 개발 기간/우선순위: 12주 full (Week 1~4 MVP 기반 마련)
```

## 출력 검토 체크포인트

1. 가정(Assumptions)과 영향이 분리되어 있는가
2. KV-cache 포맷/대역폭 계산식이 있는가
3. 레지스터 맵에 오프셋/비트필드가 포함되어 있는가
4. 성능 계산 예시와 정확도-성능 트레이드오프 표가 있는가
5. TODO 10개가 구현 순서로 정렬되어 있는가
