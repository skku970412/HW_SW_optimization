`timescale 1ns/1ps

module gemm_core #(
    parameter int M_TILE = 4,
    parameter int N_TILE = 8,
    parameter int K_TILE = 16,
    parameter int A_WIDTH = 16,
    parameter int B_WIDTH = 8,
    parameter int ACC_WIDTH = 32
) (
    input  logic clk,
    input  logic rst_n,
    input  logic cfg_clear_perf,
    input  logic [15:0] cfg_k_tile,
    input  logic in_valid,
    output logic in_ready,
    input  logic [A_WIDTH-1:0] a_data,
    input  logic [B_WIDTH-1:0] b_data,
    input  logic cfg_start,
    output logic out_valid,
    input  logic out_ready,
    output logic [ACC_WIDTH-1:0] out_data,
    output logic [31:0] perf_cycle_count,
    output logic [31:0] perf_mac_count
);
    localparam int KCNT_W = 16;
    typedef enum logic [1:0] {
        ST_IDLE = 2'd0,
        ST_RUN  = 2'd1,
        ST_OUT  = 2'd2
    } state_t;

    state_t state;
    logic signed [ACC_WIDTH-1:0] acc;
    logic [KCNT_W-1:0] issued_count;
    logic [KCNT_W-1:0] done_count;
    logic [KCNT_W-1:0] k_target;
    logic fire_in;
    logic fire_out;
    logic signed [ACC_WIDTH-1:0] product;
    logic signed [ACC_WIDTH-1:0] prod_pipe;
    logic prod_pipe_valid;
    logic consume_pipe;

    assign fire_in = in_valid && in_ready;
    assign fire_out = out_valid && out_ready;
    assign product = $signed(a_data) * $signed(b_data);
    assign consume_pipe = prod_pipe_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= ST_IDLE;
            acc       <= '0;
            issued_count <= '0;
            done_count <= '0;
            k_target <= K_TILE;
            prod_pipe <= '0;
            prod_pipe_valid <= 1'b0;
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
                        acc <= '0;
                        issued_count <= '0;
                        done_count <= '0;
                        prod_pipe_valid <= 1'b0;
                        k_target <= (cfg_k_tile == 16'd0) ? K_TILE[15:0] : cfg_k_tile;
                        state <= ST_RUN;
                    end
                end

                ST_RUN: begin
                    if (consume_pipe) begin
                        acc <= acc + prod_pipe;
                        done_count <= done_count + 1'b1;
                        perf_mac_count <= perf_mac_count + 1'b1;
                        if ((done_count + 1'b1) >= k_target) begin
                            state <= ST_OUT;
                            out_valid <= 1'b1;
                        end
                    end

                    if (fire_in) begin
                        prod_pipe <= product;
                        prod_pipe_valid <= 1'b1;
                        issued_count <= issued_count + 1'b1;
                    end else begin
                        prod_pipe_valid <= 1'b0;
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
    assign out_data = acc;
endmodule
