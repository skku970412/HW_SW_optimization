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

`ifdef KV_CACHE_USE_XPM
    // ---------------------------------------------------------------------
    // Xilinx XPM implementation
    // ---------------------------------------------------------------------
    // Why this exists:
    // - Small memories (e.g., 256x16) often get mapped to LUTRAM even with
    //   ram_style="block". For portfolio QoR, we sometimes want to *force*
    //   BRAM usage to demonstrate control over memory implementation.
    // - This path instantiates XPM RAM with MEMORY_PRIMITIVE="block".
    //
    // NOTE: KV_CACHE_USE_XPM should be defined only in Vivado QoR runs.

    localparam int ADDR_W = $clog2(DEPTH);
    localparam int WORD_W = 2 * DATA_WIDTH;

    logic [WORD_W-1:0] doutb;
    logic [0:0] wea;
    logic collision_d;
    logic [DATA_WIDTH-1:0] k_bypass_d;
    logic [DATA_WIDTH-1:0] v_bypass_d;
    assign wea[0] = wr_en;

    // Pack {K,V} into a single BRAM word.
    // doutb[WORD_W-1:DATA_WIDTH] -> K, doutb[DATA_WIDTH-1:0] -> V

    xpm_memory_sdpram #(
        .ADDR_WIDTH_A(ADDR_W),
        .ADDR_WIDTH_B(ADDR_W),
        .AUTO_SLEEP_TIME(0),
        .BYTE_WRITE_WIDTH_A(WORD_W),
        .CASCADE_HEIGHT(0),
        .CLOCKING_MODE("common_clock"),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE("none"),
        .MEMORY_INIT_PARAM("0"),
        .MEMORY_OPTIMIZATION("true"),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE(DEPTH * WORD_W),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_B(WORD_W),
        .READ_LATENCY_B(1),
        .READ_RESET_VALUE_B("0"),
        .RST_MODE_A("SYNC"),
        .RST_MODE_B("ASYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_MEM_INIT(0),
        .WAKEUP_TIME("disable_sleep"),
        .WRITE_DATA_WIDTH_A(WORD_W),
        .WRITE_MODE_B("read_first")
    ) u_kv_bram (
        .clka(clk),
        .ena(wr_en),
        .wea(wea),
        .addra(wr_addr),
        .dina({k_wr_data, v_wr_data}),
        .injectsbiterra(1'b0),
        .injectdbiterra(1'b0),
        .clkb(clk),
        .enb(rd_en),
        .addrb(rd_addr),
        .doutb(doutb),
        .sbiterrb(),
        .dbiterrb(),
        .rstb(~rst_n),
        .regceb(1'b1),
        .sleep(1'b0)
    );

    // Emulate write-first behavior for read-port collisions.
    // XPM SDPRAM block mode requires READ_FIRST on port B, so we bypass
    // the registered write payload when previous cycle had rd_en+wr_en+same_addr.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            collision_d <= 1'b0;
            k_bypass_d <= '0;
            v_bypass_d <= '0;
            k_rd_data <= '0;
            v_rd_data <= '0;
        end else begin
            collision_d <= rd_en && wr_en && (rd_addr == wr_addr);
            if (rd_en) begin
                k_bypass_d <= k_wr_data;
                v_bypass_d <= v_wr_data;
            end

            if (collision_d) begin
                k_rd_data <= k_bypass_d;
                v_rd_data <= v_bypass_d;
            end else begin
                k_rd_data <= doutb[WORD_W-1:DATA_WIDTH];
                v_rd_data <= doutb[DATA_WIDTH-1:0];
            end
        end
    end

`else
    // ---------------------------------------------------------------------
    // Portable inference implementation (works in open-source simulators)
    // ---------------------------------------------------------------------
    // Encourage BRAM inference for cache arrays on FPGA.
    (* ram_style = "block" *) logic [DATA_WIDTH-1:0] k_mem [0:DEPTH-1];
    (* ram_style = "block" *) logic [DATA_WIDTH-1:0] v_mem [0:DEPTH-1];

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
`endif
endmodule
