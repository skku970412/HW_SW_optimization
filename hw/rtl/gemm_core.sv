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
    input  logic in_valid,
    output logic in_ready,
    input  logic [A_WIDTH-1:0] a_data,
    input  logic [B_WIDTH-1:0] b_data,
    input  logic cfg_start,
    output logic out_valid,
    input  logic out_ready,
    output logic [ACC_WIDTH-1:0] out_data
);
    localparam int KCNT_W = (K_TILE <= 1) ? 1 : $clog2(K_TILE);
    typedef enum logic [1:0] {
        ST_IDLE = 2'd0,
        ST_RUN  = 2'd1,
        ST_OUT  = 2'd2
    } state_t;

    state_t state;
    logic signed [ACC_WIDTH-1:0] acc;
    logic [KCNT_W-1:0] k_count;
    logic fire_in;
    logic fire_out;
    logic signed [ACC_WIDTH-1:0] product;

    assign fire_in = in_valid && in_ready;
    assign fire_out = out_valid && out_ready;
    assign product = $signed(a_data) * $signed(b_data);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= ST_IDLE;
            acc       <= '0;
            k_count   <= '0;
            out_valid <= 1'b0;
        end else begin
            case (state)
                ST_IDLE: begin
                    out_valid <= 1'b0;
                    if (cfg_start) begin
                        acc <= '0;
                        k_count <= '0;
                        state <= ST_RUN;
                    end
                end

                ST_RUN: begin
                    if (fire_in) begin
                        acc <= acc + product;
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
    assign out_data = acc;
endmodule
