// Fixed-point Conv2-pruned LeNet-5 character classifier.
// Inputs are 32x32 reference character crops embedded in the .sof ROM.
// The JTAG source chooses a sample (bits 6:0) and pulses start (bit 7).
// Probe [63:32]=latched cycles, [31:24]=completion sequence,
// [15:8]=latched sample, [7]=result valid, [6]=busy,
// [4:0]=latched class index. LEDR0 latches done.
module OcrDemo (
    input  wire MAX10_CLK1_50,
    output wire LEDR0
);
    localparam integer NUM_SAMPLES = 50;
    localparam integer NUM_WEIGHTS = 37950;
    localparam integer NUM_BIASES = 248;
    localparam [7:0] S_IDLE = 0, S_INIT = 1, S_READ = 2, S_MAC = 3,
                     S_ACT = 4, S_ADVANCE = 5, S_POOL_READ = 6,
                     S_POOL_ACC = 7, S_POOL_WRITE = 8;

    reg signed [7:0] weights [0:NUM_WEIGHTS-1];
    reg signed [31:0] biases [0:NUM_BIASES-1];
    reg signed [7:0] images [0:NUM_SAMPLES*1024-1];
    reg signed [7:0] tanh_lut [0:256];
    reg signed [7:0] act_a [0:4703];
    reg signed [7:0] act_b [0:1175];
    initial begin
        $readmemh("generated/weights.hex", weights);
        $readmemh("generated/biases.hex", biases);
        $readmemh("generated/images.hex", images);
        $readmemh("generated/tanh.hex", tanh_lut);
    end

    wire [7:0] jtag_source;
    wire [63:0] jtag_probe;
    altsource_probe #(
        .sld_auto_instance_index("YES"),
        .sld_instance_index(0),
        .instance_id("OCR0"),
        .probe_width(64),
        .source_width(8),
        .source_initial_value("0"),
        .enable_metastability("NO")
    ) probe0 (
        .source(jtag_source),
        .probe(jtag_probe),
        .source_clk(),
        .source_ena()
    );

    reg [7:0] source_meta = 0, source_sync = 0;

    reg [7:0] state = S_IDLE;
    reg [2:0] layer = 0;
    reg [12:0] out_idx = 0;
    reg [7:0] tap_idx = 0;
    reg [4:0] ox = 0, oy = 0;
    reg [3:0] oc = 0, ic = 0;
    reg [2:0] kx = 0, ky = 0;
    reg [10:0] pool_idx = 0;
    reg [3:0] poc = 0;
    reg [3:0] px = 0, py = 0;
    reg [1:0] pool_tap = 0;
    reg signed [10:0] pool_sum = 0;
    reg signed [7:0] w_data = 0, x_data = 0;
    reg signed [7:0] a_data = 0, b_data = 0;
    reg signed [31:0] acc = 0;
    reg signed [31:0] best_score = 32'sh80000000;
    reg [4:0] pred = 0;
    reg [6:0] sample_id = 0;
    reg done = 0, busy = 0, start_prev = 0;
    reg [31:0] cycles = 0;
    reg [31:0] result_cycles = 0;
    reg [7:0] result_seq = 0;
    reg [6:0] result_sample = 0;
    reg [4:0] result_pred = 0;
    reg result_valid = 0;

    assign LEDR0 = done;
    assign jtag_probe = {result_cycles, result_seq, 8'd0, 1'b0, result_sample,
                         result_valid, busy, 1'b0, result_pred};

    wire [15:0] weight_base = layer == 0 ? 16'd0 :
                              layer == 1 ? 16'd150 :
                              layer == 2 ? 16'd1350 :
                              layer == 3 ? 16'd25350 : 16'd35430;
    wire [7:0] bias_base = layer == 0 ? 8'd0 :
                           layer == 1 ? 8'd6 :
                           layer == 2 ? 8'd14 :
                           layer == 3 ? 8'd134 : 8'd218;
    wire [7:0] tap_limit = layer == 0 ? 8'd25 :
                            layer == 1 ? 8'd150 :
                            layer == 2 ? 8'd200 :
                            layer == 3 ? 8'd120 : 8'd84;
    wire [12:0] out_limit = layer == 0 ? 13'd4704 :
                             layer == 1 ? 13'd800 :
                             layer == 2 ? 13'd120 :
                             layer == 3 ? 13'd84 : 13'd30;
    wire [15:0] weight_addr = weight_base +
        (layer == 0 ? oc * 16'd25 :
         layer == 1 ? oc * 16'd150 :
         layer == 2 ? out_idx * 16'd200 :
         layer == 3 ? out_idx * 16'd120 : out_idx * 16'd84) + tap_idx;
    wire [7:0] bias_addr = bias_base + (layer < 2 ? oc : out_idx[7:0]);
    wire [16:0] image_addr = sample_id * 17'd1024 +
                             (oy + ky) * 17'd32 + ox + kx;
    wire [12:0] conv2_input_addr = ic * 13'd196 +
                                   (oy + ky) * 13'd14 + ox + kx;
    wire [12:0] pool_input_addr = layer == 0 ?
        poc * 13'd784 + (py * 2 + pool_tap[1]) * 13'd28 + px * 2 + pool_tap[0] :
        poc * 13'd100 + (py * 2 + pool_tap[1]) * 13'd10 + px * 2 + pool_tap[0];
    wire [12:0] a_read_addr = state == S_POOL_READ ? pool_input_addr : tap_idx;
    wire [12:0] b_read_addr = layer == 1 ? conv2_input_addr : tap_idx;
    wire [4:0] shift_amount = layer == 2 ? 5'd10 : 5'd9;
    wire signed [31:0] tanh_index = acc >>> shift_amount;
    wire [8:0] lut_addr = tanh_index < -32'sd128 ? 9'd0 :
                           tanh_index > 32'sd128 ? 9'd256 : tanh_index + 32'sd128;

    always @(posedge MAX10_CLK1_50) begin
        source_meta <= jtag_source;
        source_sync <= source_meta;
        // One registered read port for each activation RAM, independent of FSM branch.
        a_data <= act_a[a_read_addr];
        b_data <= act_b[b_read_addr];
        start_prev <= source_sync[7];
        if (busy) cycles <= cycles + 1'd1;
        case (state)
            S_IDLE: begin
                if (source_sync[7] && (!start_prev || source_sync[6:0] != sample_id)
                    && source_sync[6:0] < NUM_SAMPLES) begin
                    sample_id <= source_sync[6:0];
                    layer <= 0;
                    out_idx <= 0;
                    oc <= 0; ox <= 0; oy <= 0;
                    pred <= 0;
                    best_score <= 32'sh80000000;
                    cycles <= 0;
                    done <= 0;
                    busy <= 1;
                    result_valid <= 0;
                    state <= S_INIT;
                end
            end
            S_INIT: begin
                acc <= biases[bias_addr];
                tap_idx <= 0;
                kx <= 0; ky <= 0; ic <= 0;
                state <= S_READ;
            end
            S_READ: begin
                w_data <= weights[weight_addr];
                if (layer == 0) x_data <= images[image_addr];
                state <= S_MAC;
            end
            S_MAC: begin
                acc <= acc + $signed(w_data) *
                    $signed(layer == 0 ? x_data : layer == 3 ? a_data : b_data);
                if (tap_idx + 1'b1 == tap_limit) begin
                    state <= S_ACT;
                end else begin
                    tap_idx <= tap_idx + 1'b1;
                    if (layer < 2) begin
                        if (kx == 4) begin
                            kx <= 0;
                            if (ky == 4) begin
                                ky <= 0;
                                ic <= ic + 1'b1;
                            end else ky <= ky + 1'b1;
                        end else kx <= kx + 1'b1;
                    end
                    state <= S_READ;
                end
            end
            S_ACT: begin
                if (layer == 4) begin
                    if (acc > best_score) begin
                        best_score <= acc;
                        pred <= out_idx[4:0];
                    end
                end else if (layer == 3) begin
                    act_b[out_idx] <= tanh_lut[lut_addr];
                end else begin
                    act_a[out_idx] <= tanh_lut[lut_addr];
                end
                state <= S_ADVANCE;
            end
            S_ADVANCE: begin
                if (out_idx + 1'b1 == out_limit) begin
                    if (layer == 0 || layer == 1) begin
                        pool_idx <= 0;
                        poc <= 0; px <= 0; py <= 0;
                        pool_tap <= 0;
                        pool_sum <= 0;
                        state <= S_POOL_READ;
                    end else if (layer == 4) begin
                        busy <= 0;
                        done <= 1;
                        result_cycles <= cycles + 1'b1;
                        result_sample <= sample_id;
                        result_pred <= pred;
                        result_seq <= result_seq + 1'b1;
                        result_valid <= 1;
                        state <= S_IDLE;
                    end else begin
                        layer <= layer + 1'b1;
                        out_idx <= 0;
                        state <= S_INIT;
                    end
                end else begin
                    out_idx <= out_idx + 1'b1;
                    if (layer < 2) begin
                        if ((layer == 0 && ox == 27) || (layer == 1 && ox == 9)) begin
                            ox <= 0;
                            if ((layer == 0 && oy == 27) || (layer == 1 && oy == 9)) begin
                                oy <= 0;
                                oc <= oc + 1'b1;
                            end else oy <= oy + 1'b1;
                        end else ox <= ox + 1'b1;
                    end
                    state <= S_INIT;
                end
            end
            S_POOL_READ: begin
                state <= S_POOL_ACC;
            end
            S_POOL_ACC: begin
                pool_sum <= pool_sum + $signed(a_data);
                if (pool_tap == 3) state <= S_POOL_WRITE;
                else begin
                    pool_tap <= pool_tap + 1'b1;
                    state <= S_POOL_READ;
                end
            end
            S_POOL_WRITE: begin
                act_b[pool_idx] <= (pool_sum + 11'sd2) >>> 2;
                if ((layer == 0 && pool_idx == 1175) ||
                    (layer == 1 && pool_idx == 199)) begin
                    layer <= layer + 1'b1;
                    out_idx <= 0;
                    oc <= 0; ox <= 0; oy <= 0;
                    state <= S_INIT;
                end else begin
                    pool_idx <= pool_idx + 1'b1;
                    if ((layer == 0 && px == 13) || (layer == 1 && px == 4)) begin
                        px <= 0;
                        if ((layer == 0 && py == 13) || (layer == 1 && py == 4)) begin
                            py <= 0;
                            poc <= poc + 1'b1;
                        end else py <= py + 1'b1;
                    end else px <= px + 1'b1;
                    pool_tap <= 0;
                    pool_sum <= 0;
                    state <= S_POOL_READ;
                end
            end
            default: state <= S_IDLE;
        endcase
    end
endmodule
