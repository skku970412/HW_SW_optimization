`timescale 1ns/1ps

module kv_cache #(
    parameter int DATA_WIDTH = 16,
    parameter int DEPTH = 256
) (
    input  logic clk,
    input  logic rst_n,
    input  logic wr_en,
    input  logic [$clog2(DEPTH)-1:0] wr_addr,
    input  logic [DATA_WIDTH-1:0] k_wr_data,
    input  logic [DATA_WIDTH-1:0] v_wr_data,
    input  logic rd_en,
    input  logic [$clog2(DEPTH)-1:0] rd_addr,
    output logic [DATA_WIDTH-1:0] k_rd_data,
    output logic [DATA_WIDTH-1:0] v_rd_data
);
    logic [DATA_WIDTH-1:0] k_mem [0:DEPTH-1];
    logic [DATA_WIDTH-1:0] v_mem [0:DEPTH-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            k_rd_data <= '0;
            v_rd_data <= '0;
        end else begin
            if (wr_en) begin
                k_mem[wr_addr] <= k_wr_data;
                v_mem[wr_addr] <= v_wr_data;
            end

            if (rd_en) begin
                // Write-first behavior on same-address collision.
                if (wr_en && (wr_addr == rd_addr)) begin
                    k_rd_data <= k_wr_data;
                    v_rd_data <= v_wr_data;
                end else begin
                    k_rd_data <= k_mem[rd_addr];
                    v_rd_data <= v_mem[rd_addr];
                end
            end
        end
    end
endmodule
