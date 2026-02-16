module attention_core #(
    parameter int DATA_WIDTH = 16,
    parameter int SCORE_WIDTH = 24
) (
    input  logic clk,
    input  logic rst_n,
    input  logic in_valid,
    output logic in_ready,
    input  logic [DATA_WIDTH-1:0] q_data,
    input  logic [DATA_WIDTH-1:0] k_data,
    input  logic [DATA_WIDTH-1:0] v_data,
    output logic out_valid,
    input  logic out_ready,
    output logic [DATA_WIDTH-1:0] out_data
);
    // MVP skeleton. Replace with QK^T, softmax approximation, and AV pipeline.
    logic [SCORE_WIDTH-1:0] score_acc;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            score_acc <= '0;
            out_valid <= 1'b0;
        end else begin
            if (in_valid && in_ready) begin
                score_acc <= score_acc + ($signed(q_data) * $signed(k_data));
                out_valid <= 1'b1;
            end
            if (out_valid && out_ready) begin
                out_valid <= 1'b0;
            end
        end
    end

    assign in_ready = 1'b1;
    assign out_data = v_data;
endmodule
