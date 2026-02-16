`timescale 1ns/1ps

module decoder_block_top #(
    parameter int DATA_WIDTH = 16
) (
    input  logic clk,
    input  logic rst_n,
    input  logic cfg_start,
    input  logic in_valid,
    output logic in_ready,
    input  logic [DATA_WIDTH-1:0] x_t_data,
    output logic out_valid,
    input  logic out_ready,
    output logic [DATA_WIDTH-1:0] y_t_data
);
    logic attn_valid;
    logic [DATA_WIDTH-1:0] attn_data;

    attention_core #(
        .DATA_WIDTH(DATA_WIDTH)
    ) u_attention (
        .clk(clk),
        .rst_n(rst_n),
        .cfg_start(cfg_start),
        .cfg_clear_perf(1'b0),
        .cfg_k_tile(16'd0),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .q_data(x_t_data),
        .k_data(x_t_data),
        .v_data(x_t_data),
        .out_valid(attn_valid),
        .out_ready(out_ready),
        .out_data(attn_data),
        .perf_cycle_count(),
        .perf_mac_count()
    );

    assign out_valid = attn_valid;
    assign y_t_data  = attn_data;
endmodule
