# RTL TB 실행 가이드 (시뮬레이터 설치 후)

## 전제

1. RTL 시뮬레이터 중 하나가 PATH에 있어야 한다.
2. 현재 스모크 대상 DUT: `hw/rtl/gemm_core.sv`
3. cocotb 테스트 파일: `tests/tb/test_gemm_cocotb.py`

## 예시 (Icarus/Verilator 환경에서 확장)

실제 실행 명령은 사용하는 시뮬레이터에 맞게 아래 변수를 설정해 사용한다.

1. `TOPLEVEL=gemm_core`
2. `VERILOG_SOURCES=hw/rtl/gemm_core.sv`
3. `MODULE=test_gemm_cocotb`

현재 환경은 시뮬레이터가 없어 `tests/tb/test_simulator_gate.py`에서 BLOCKED가 기록된다.
