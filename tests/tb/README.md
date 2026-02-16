# RTL TB 실행 가이드 (시뮬레이터 설치 후)

## 전제

1. RTL 시뮬레이터 중 하나가 PATH에 있어야 한다.
2. 현재 스모크 대상 DUT:
- `hw/rtl/gemm_core.sv`
- `hw/rtl/attention_core.sv`
- `hw/rtl/kv_cache.sv`
- `hw/rtl/npu_top.sv`
3. cocotb 테스트 파일:
- `tests/tb/cocotb_gemm.py`
- `tests/tb/cocotb_attention.py`
- `tests/tb/cocotb_kvcache.py`
- `tests/tb/cocotb_npu_top.py`

## 예시 (Icarus/Verilator 환경에서 확장)

실제 실행 명령은 사용하는 시뮬레이터에 맞게 아래 변수를 설정해 사용한다.

1. `TOPLEVEL=npu_top`
2. `VERILOG_SOURCES=hw/rtl/npu_top.sv`
3. `MODULE=tests.tb.cocotb_npu_top`

자동 실행 명령(권장):

```powershell
python -m pytest tests/tb/test_rtl_compile.py tests/tb/test_cocotb_runner.py -q
```
