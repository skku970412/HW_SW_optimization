`timescale 1ns/1ps

module attention_core #(
    parameter int DATA_WIDTH = 16,
    parameter int SCORE_WIDTH = 24,
    parameter int K_TILE = 16,
    parameter int VALUE_SHIFT = 8
) (
    input  logic clk,
    input  logic rst_n,
    input  logic cfg_start,
    input  logic in_valid,
    output logic in_ready,
    input  logic [DATA_WIDTH-1:0] q_data,
    input  logic [DATA_WIDTH-1:0] k_data,
    input  logic [DATA_WIDTH-1:0] v_data,
    output logic out_valid,
    input  logic out_ready,
    output logic [DATA_WIDTH-1:0] out_data
);
    localparam int KCNT_W = (K_TILE <= 1) ? 1 : $clog2(K_TILE);
    localparam int QK_W = (SCORE_WIDTH > (2 * DATA_WIDTH)) ? SCORE_WIDTH : (2 * DATA_WIDTH);
    localparam int VALUE_ACC_W = QK_W + DATA_WIDTH;

    typedef enum logic [1:0] {
        ST_IDLE = 2'd0,
        ST_RUN  = 2'd1,
        ST_OUT  = 2'd2
    } state_t;

    state_t state;
    logic [KCNT_W-1:0] k_count;

    logic signed [QK_W-1:0] score_acc;
    logic signed [VALUE_ACC_W-1:0] value_acc;
    logic signed [QK_W-1:0] qk_mul;
    logic signed [VALUE_ACC_W-1:0] qkv_mul;
    logic signed [VALUE_ACC_W-1:0] out_shifted;
    logic fire_in;
    logic fire_out;

    assign fire_in = in_valid && in_ready;
    assign fire_out = out_valid && out_ready;
    assign qk_mul = $signed(q_data) * $signed(k_data);
    assign qkv_mul = qk_mul * $signed(v_data);
    assign out_shifted = value_acc >>> VALUE_SHIFT;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            k_count <= '0;
            score_acc <= '0;
            value_acc <= '0;
            out_valid <= 1'b0;
        end else begin
            case (state)
                ST_IDLE: begin
                    out_valid <= 1'b0;
                    if (cfg_start) begin
                        score_acc <= '0;
                        value_acc <= '0;
                        k_count <= '0;
                        state <= ST_RUN;
                    end
                end

                ST_RUN: begin
                    if (fire_in) begin
                        score_acc <= score_acc + qk_mul;
                        value_acc <= value_acc + qkv_mul;
                        if (k_count == K_TILE - 1) begin
                            state <= ST_OUT;
                            out_valid <= 1'b1;
                        end else begin
                            k_count <= k_count + 1'b1;
                        end
                    end
                end

                ST_OUT: begin
                    if (fire_out) begin
                        out_valid <= 1'b0;
                        state <= ST_IDLE;
                    end
                end

                default: begin
                    state <= ST_IDLE;
                    out_valid <= 1'b0;
                end
            endcase
        end
    end

    assign in_ready = (state == ST_RUN);
    assign out_data = out_shifted[DATA_WIDTH-1:0];
endmodule
