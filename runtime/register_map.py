from __future__ import annotations

# Simple MMIO-style register map for boardless runtime.
REG_CONTROL = 0x00
REG_STATUS = 0x04
REG_PROMPT_LEN = 0x08
REG_GEN_LEN = 0x0C
REG_DONE_TOKENS = 0x10
REG_LAST_ERROR = 0x14

CTRL_START = 1 << 0
CTRL_RESET = 1 << 1

STATUS_BUSY = 1 << 0
STATUS_DONE = 1 << 1
STATUS_ERROR = 1 << 2
