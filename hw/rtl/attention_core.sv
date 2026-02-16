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
    input  logic cfg_clear_perf,
    input  logic [15:0] cfg_k_tile,
    input  logic in_valid,
    output logic in_ready,
    input  logic [DATA_WIDTH-1:0] q_data,
    input  logic [DATA_WIDTH-1:0] k_data,
    input  logic [DATA_WIDTH-1:0] v_data,
    output logic out_valid,
    input  logic out_ready,
    output logic [DATA_WIDTH-1:0] out_data,
    output logic [31:0] perf_cycle_count,
    output logic [31:0] perf_mac_count
);
    localparam int KCNT_W = 16;
    localparam int QK_W = (SCORE_WIDTH > (2 * DATA_WIDTH)) ? SCORE_WIDTH : (2 * DATA_WIDTH);
    localparam int VALUE_ACC_W = QK_W + DATA_WIDTH;

    typedef enum logic [1:0] {
        ST_IDLE = 2'd0,
        ST_RUN  = 2'd1,
        ST_OUT  = 2'd2
    } state_t;

    state_t state;
    logic [KCNT_W-1:0] issued_count;
    logic [KCNT_W-1:0] done_count;
    logic [KCNT_W-1:0] k_target;

    logic signed [QK_W-1:0] score_acc;
    logic signed [VALUE_ACC_W-1:0] value_acc;
    logic signed [QK_W-1:0] qk_mul;
    logic signed [VALUE_ACC_W-1:0] qkv_mul;
    logic signed [VALUE_ACC_W-1:0] out_shifted;
    logic signed [QK_W-1:0] qk_pipe;
    logic signed [VALUE_ACC_W-1:0] qkv_pipe;
    logic pipe_valid;
    logic fire_in;
    logic fire_out;
    logic consume_pipe;

    assign fire_in = in_valid && in_ready;
    assign fire_out = out_valid && out_ready;
    assign consume_pipe = pipe_valid;
    assign qk_mul = $signed(q_data) * $signed(k_data);
    assign qkv_mul = qk_mul * $signed(v_data);
    assign out_shifted = value_acc >>> VALUE_SHIFT;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            issued_count <= '0;
            done_count <= '0;
            k_target <= K_TILE;
            score_acc <= '0;
            value_acc <= '0;
            qk_pipe <= '0;
            qkv_pipe <= '0;
            pipe_valid <= 1'b0;
            out_valid <= 1'b0;
            perf_cycle_count <= '0;
            perf_mac_count <= '0;
        end else begin
            if (cfg_clear_perf) begin
                perf_cycle_count <= '0;
                perf_mac_count <= '0;
            end
            if (state != ST_IDLE) begin
                perf_cycle_count <= perf_cycle_count + 1'b1;
            end

            case (state)
                ST_IDLE: begin
                    out_valid <= 1'b0;
                    if (cfg_start) begin
                        score_acc <= '0;
                        value_acc <= '0;
                        issued_count <= '0;
                        done_count <= '0;
                        pipe_valid <= 1'b0;
                        k_target <= (cfg_k_tile == 16'd0) ? K_TILE[15:0] : cfg_k_tile;
                        state <= ST_RUN;
                    end
                end

                ST_RUN: begin
                    if (consume_pipe) begin
                        score_acc <= score_acc + qk_pipe;
                        value_acc <= value_acc + qkv_pipe;
                        done_count <= done_count + 1'b1;
                        perf_mac_count <= perf_mac_count + 1'b1;
                        if ((done_count + 1'b1) >= k_target) begin
                            state <= ST_OUT;
                            out_valid <= 1'b1;
                        end
                    end

                    if (fire_in) begin
                        qk_pipe <= qk_mul;
                        qkv_pipe <= qkv_mul;
                        pipe_valid <= 1'b1;
                        issued_count <= issued_count + 1'b1;
                    end else begin
                        pipe_valid <= 1'b0;
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

    assign in_ready = (state == ST_RUN) && (issued_count < k_target);
    assign out_data = out_shifted[DATA_WIDTH-1:0];
endmodule
