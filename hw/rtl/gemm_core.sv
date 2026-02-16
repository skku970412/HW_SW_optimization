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
    // MVP skeleton. Replace with tiled MAC datapath in Week B2/B3.
    logic busy;
    logic [ACC_WIDTH-1:0] acc;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy      <= 1'b0;
            acc       <= '0;
            out_valid <= 1'b0;
        end else begin
            if (cfg_start) begin
                busy      <= 1'b1;
                acc       <= '0;
                out_valid <= 1'b0;
            end
            if (busy && in_valid && in_ready) begin
                acc <= acc + $signed(a_data) * $signed(b_data);
            end
            if (busy && out_ready) begin
                out_valid <= 1'b1;
                busy      <= 1'b0;
            end
        end
    end

    assign in_ready = busy;
    assign out_data = acc;
endmodule
