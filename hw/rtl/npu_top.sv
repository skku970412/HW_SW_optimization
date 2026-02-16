`timescale 1ns/1ps

module npu_top #(
    parameter int ADDR_WIDTH = 8,
    parameter int TOKEN_LATENCY = 32
) (
    input  logic clk,
    input  logic rst_n,
    input  logic mmio_wr_en,
    input  logic mmio_rd_en,
    input  logic [ADDR_WIDTH-1:0] mmio_addr,
    input  logic [31:0] mmio_wdata,
    output logic [31:0] mmio_rdata,
    output logic mmio_ready
);
    localparam logic [ADDR_WIDTH-1:0] ADDR_CONTROL       = 8'h00;
    localparam logic [ADDR_WIDTH-1:0] ADDR_STATUS        = 8'h04;
    localparam logic [ADDR_WIDTH-1:0] ADDR_PROMPT_LEN    = 8'h08;
    localparam logic [ADDR_WIDTH-1:0] ADDR_GEN_LEN       = 8'h0C;
    localparam logic [ADDR_WIDTH-1:0] ADDR_DONE_TOKENS   = 8'h10;
    localparam logic [ADDR_WIDTH-1:0] ADDR_LAST_ERROR    = 8'h14;
    localparam logic [ADDR_WIDTH-1:0] ADDR_PERF_CYCLES   = 8'h18;
    localparam logic [ADDR_WIDTH-1:0] ADDR_PERF_TOKENS   = 8'h1C;
    localparam logic [ADDR_WIDTH-1:0] ADDR_PERF_STALL_IN = 8'h20;
    localparam logic [ADDR_WIDTH-1:0] ADDR_PERF_STALL_OUT= 8'h24;
    localparam logic [ADDR_WIDTH-1:0] ADDR_CFG_K_TILE    = 8'h28;

    localparam int CTRL_START_BIT = 0;
    localparam int CTRL_RESET_BIT = 1;

    localparam logic [31:0] STATUS_BUSY  = 32'h0000_0001;
    localparam logic [31:0] STATUS_DONE  = 32'h0000_0002;
    localparam logic [31:0] STATUS_ERROR = 32'h0000_0004;

    logic [31:0] reg_control;
    logic [31:0] reg_status;
    logic [31:0] reg_prompt_len;
    logic [31:0] reg_gen_len;
    logic [31:0] reg_done_tokens;
    logic [31:0] reg_last_error;
    logic [31:0] reg_perf_cycles;
    logic [31:0] reg_perf_tokens;
    logic [31:0] reg_perf_stall_in;
    logic [31:0] reg_perf_stall_out;
    logic [31:0] reg_cfg_k_tile;

    logic run_active;
    logic [31:0] token_cycle;
    logic [31:0] token_target_cycles;
    logic start_evt;
    logic reset_evt;

    assign start_evt = mmio_wr_en && (mmio_addr == ADDR_CONTROL) && mmio_wdata[CTRL_START_BIT];
    assign reset_evt = mmio_wr_en && (mmio_addr == ADDR_CONTROL) && mmio_wdata[CTRL_RESET_BIT];
    assign mmio_ready = mmio_wr_en || mmio_rd_en;
    assign token_target_cycles = TOKEN_LATENCY
        + ((reg_cfg_k_tile < 32'd4) ? 32'd10 : 32'd0)
        + ((reg_cfg_k_tile < 32'd8) ? 32'd6 : 32'd0)
        + ((reg_done_tokens[7:0]) >> 2);

    always_comb begin
        unique case (mmio_addr)
            ADDR_CONTROL:        mmio_rdata = reg_control;
            ADDR_STATUS:         mmio_rdata = reg_status;
            ADDR_PROMPT_LEN:     mmio_rdata = reg_prompt_len;
            ADDR_GEN_LEN:        mmio_rdata = reg_gen_len;
            ADDR_DONE_TOKENS:    mmio_rdata = reg_done_tokens;
            ADDR_LAST_ERROR:     mmio_rdata = reg_last_error;
            ADDR_PERF_CYCLES:    mmio_rdata = reg_perf_cycles;
            ADDR_PERF_TOKENS:    mmio_rdata = reg_perf_tokens;
            ADDR_PERF_STALL_IN:  mmio_rdata = reg_perf_stall_in;
            ADDR_PERF_STALL_OUT: mmio_rdata = reg_perf_stall_out;
            ADDR_CFG_K_TILE:     mmio_rdata = reg_cfg_k_tile;
            default:             mmio_rdata = 32'h0000_0000;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_control <= 32'h0;
            reg_status <= 32'h0;
            reg_prompt_len <= 32'h0;
            reg_gen_len <= 32'h0;
            reg_done_tokens <= 32'h0;
            reg_last_error <= 32'h0;
            reg_perf_cycles <= 32'h0;
            reg_perf_tokens <= 32'h0;
            reg_perf_stall_in <= 32'h0;
            reg_perf_stall_out <= 32'h0;
            reg_cfg_k_tile <= 32'd16;
            run_active <= 1'b0;
            token_cycle <= 32'h0;
        end else begin
            if (mmio_wr_en) begin
                unique case (mmio_addr)
                    ADDR_CONTROL: reg_control <= mmio_wdata;
                    ADDR_PROMPT_LEN: reg_prompt_len <= mmio_wdata;
                    ADDR_GEN_LEN: reg_gen_len <= mmio_wdata;
                    ADDR_CFG_K_TILE: reg_cfg_k_tile <= mmio_wdata;
                    default: begin end
                endcase
            end

            if (reset_evt) begin
                reg_status <= 32'h0;
                reg_done_tokens <= 32'h0;
                reg_last_error <= 32'h0;
                reg_perf_cycles <= 32'h0;
                reg_perf_tokens <= 32'h0;
                reg_perf_stall_in <= 32'h0;
                reg_perf_stall_out <= 32'h0;
                run_active <= 1'b0;
                token_cycle <= 32'h0;
            end else begin
                if (start_evt) begin
                    reg_done_tokens <= 32'h0;
                    reg_last_error <= 32'h0;
                    reg_perf_cycles <= 32'h0;
                    reg_perf_tokens <= 32'h0;
                    reg_perf_stall_in <= 32'h0;
                    reg_perf_stall_out <= 32'h0;
                    token_cycle <= 32'h0;

                    if ((reg_prompt_len == 32'h0) || (reg_gen_len == 32'h0)) begin
                        reg_status <= STATUS_ERROR;
                        reg_last_error <= 32'h1;
                        run_active <= 1'b0;
                    end else begin
                        reg_status <= STATUS_BUSY;
                        run_active <= 1'b1;
                    end
                end

                if (run_active) begin
                    reg_perf_cycles <= reg_perf_cycles + 1'b1;

                    if ((token_cycle + 1'b1) >= token_target_cycles) begin
                        token_cycle <= 32'h0;
                        reg_done_tokens <= reg_done_tokens + 1'b1;
                        reg_perf_tokens <= reg_perf_tokens + 1'b1;

                        if ((reg_done_tokens + 1'b1) >= reg_gen_len) begin
                            run_active <= 1'b0;
                            reg_status <= STATUS_DONE;
                        end
                    end else begin
                        token_cycle <= token_cycle + 1'b1;
                    end

                    if (reg_cfg_k_tile < 32'd4) begin
                        reg_perf_stall_in <= reg_perf_stall_in + 1'b1;
                    end
                    if ((reg_done_tokens != 32'h0) && (reg_done_tokens[2:0] == 3'b000)) begin
                        reg_perf_stall_out <= reg_perf_stall_out + 1'b1;
                    end
                end
            end
        end
    end
endmodule
