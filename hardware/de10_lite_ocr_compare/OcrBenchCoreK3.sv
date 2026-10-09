// Matched dense/pruned fixed-point LeNet-style 3x3 ablation core.
// Stream mode loads one 32x32 character into a 1 KiB RAM in 64 chunks of 16 bytes.
// Streaming here means frame input; convolution still uses a serial MAC engine.
module OcrBenchCoreK3 #(
    parameter integer CONV2_CHANNELS = 8,
    parameter integer STREAM_INPUT = 0,
    parameter integer NUM_SAMPLES = 50
) (
    input  wire MAX10_CLK1_50,
    output wire LEDR0
);
    // 32 -> Conv3 30 -> Pool2 15 -> Conv3 13 -> Pool2 6.
    // FC1 fan-in is CONV2_CHANNELS*6*6, not the 5x5 model's *5*5.
    localparam integer NUM_WEIGHTS = 12654 + 4374 * CONV2_CHANNELS;
    localparam integer NUM_BIASES = 240 + CONV2_CHANNELS;
    localparam integer SOURCE_WIDTH = STREAM_INPUT ? 144 : 8;
    localparam [7:0] S_IDLE = 0, S_INIT = 1, S_READ = 2, S_MAC = 3,
                     S_ACT = 4, S_ADVANCE = 5, S_POOL_READ = 6,
                     S_POOL_ACC = 7, S_POOL_WRITE = 8, S_LOAD = 9;

    reg signed [7:0] weights [0:NUM_WEIGHTS-1];
    reg signed [31:0] biases [0:NUM_BIASES-1];
    reg signed [7:0] images [0:(STREAM_INPUT ? 1024 : NUM_SAMPLES*1024)-1];
    reg signed [7:0] tanh_lut [0:256];
    reg [4:0] shifts [0:4];
    reg signed [7:0] act_a [0:5399];
    reg signed [7:0] act_b [0:1349];
    initial begin
        $readmemh("generated/weights.hex", weights);
        $readmemh("generated/biases.hex", biases);
        $readmemh("generated/tanh.hex", tanh_lut);
        $readmemh("generated/shifts.hex", shifts);
    end
    generate if (STREAM_INPUT == 0) begin : image_rom_initial
        initial $readmemh("generated/images.hex", images);
    end endgenerate

    wire [SOURCE_WIDTH-1:0] jtag_source;
    wire [63:0] jtag_probe;
    altsource_probe #(
        .sld_auto_instance_index("YES"),
        .sld_instance_index(0),
        .instance_id("OCR0"),
        .probe_width(64),
        .source_width(SOURCE_WIDTH),
        .source_initial_value("0"),
        .enable_metastability("NO")
    ) probe0 (
        .source(jtag_source),
        .probe(jtag_probe),
        .source_clk(),
        .source_ena()
    );

    reg [SOURCE_WIDTH-1:0] source_meta = 0, source_sync = 0;
    wire [143:0] source_padded = source_sync;
    wire start_bit = STREAM_INPUT ? source_padded[135] : source_padded[7];
    wire [6:0] requested_sample = STREAM_INPUT ? source_padded[142:136] : source_padded[6:0];
    wire [5:0] chunk_index = source_padded[133:128];
    wire write_toggle_bit = source_padded[134];
    reg write_toggle_seen = 0;
    reg [127:0] payload_latch = 0;
    reg [5:0] chunk_latch = 0;
    reg [3:0] write_pos = 0;
    reg [6:0] chunks_loaded = 0;
    reg load_busy = 0;

    reg [7:0] state = S_IDLE;
    reg [2:0] layer = 0;
    reg [12:0] out_idx = 0;
    reg [9:0] tap_idx = 0;
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
    assign jtag_probe = {result_cycles, result_seq, 1'b0, chunks_loaded,
                         1'b0, result_sample, result_valid, busy, load_busy, result_pred};

    localparam integer W2 = 54 + 54 * CONV2_CHANNELS;
    localparam integer W3 = W2 + 4320 * CONV2_CHANNELS;
    localparam integer W4 = W3 + 10080;
    wire [16:0] weight_base = layer == 0 ? 17'd0 :
                              layer == 1 ? 17'd54 :
                              layer == 2 ? W2 :
                              layer == 3 ? W3 : W4;
    wire [7:0] bias_base = layer == 0 ? 8'd0 :
                           layer == 1 ? 8'd6 :
                           layer == 2 ? 6 + CONV2_CHANNELS :
                           layer == 3 ? 126 + CONV2_CHANNELS : 210 + CONV2_CHANNELS;
    wire [9:0] tap_limit = layer == 0 ? 10'd9 :
                            layer == 1 ? 10'd54 :
                            layer == 2 ? 36 * CONV2_CHANNELS :
                            layer == 3 ? 10'd120 : 10'd84;
    wire [12:0] out_limit = layer == 0 ? 13'd5400 :
                             layer == 1 ? 169 * CONV2_CHANNELS :
                             layer == 2 ? 13'd120 :
                             layer == 3 ? 13'd84 : 13'd30;
    wire [16:0] weight_addr = weight_base +
        (layer == 0 ? oc * 17'd9 :
         layer == 1 ? oc * 17'd54 :
         layer == 2 ? out_idx * (36 * CONV2_CHANNELS) :
         layer == 3 ? out_idx * 17'd120 : out_idx * 17'd84) + tap_idx;
    wire [7:0] bias_addr = bias_base + (layer < 2 ? oc : out_idx[7:0]);
    wire [16:0] image_addr = (STREAM_INPUT ? 17'd0 : sample_id * 17'd1024) +
                             (oy + ky) * 17'd32 + ox + kx;
    wire [12:0] conv2_input_addr = ic * 13'd225 +
                                   (oy + ky) * 13'd15 + ox + kx;
    wire [12:0] pool_input_addr = layer == 0 ?
        poc * 13'd900 + (py * 2 + pool_tap[1]) * 13'd30 + px * 2 + pool_tap[0] :
        poc * 13'd169 + (py * 2 + pool_tap[1]) * 13'd13 + px * 2 + pool_tap[0];
    wire [12:0] a_read_addr = state == S_POOL_READ ? pool_input_addr : tap_idx;
    wire [12:0] b_read_addr = layer == 1 ? conv2_input_addr : tap_idx;
    wire [4:0] shift_amount = shifts[layer];
    wire signed [31:0] tanh_index = acc >>> shift_amount;
    wire [8:0] lut_addr = tanh_index < -32'sd128 ? 9'd0 :
                           tanh_index > 32'sd128 ? 9'd256 : tanh_index + 32'sd128;

    always @(posedge MAX10_CLK1_50) begin
        source_meta <= jtag_source;
        source_sync <= source_meta;
        // One registered read port for each activation RAM, independent of FSM branch.
        a_data <= act_a[a_read_addr];
        b_data <= act_b[b_read_addr];
        start_prev <= start_bit;
        if (busy) cycles <= cycles + 1'd1;
        case (state)
            S_IDLE: begin
                if (STREAM_INPUT && write_toggle_bit != write_toggle_seen &&
                    chunks_loaded < 64 && chunk_index == chunks_loaded[5:0]) begin
                    write_toggle_seen <= write_toggle_bit;
                    payload_latch <= source_padded[127:0];
                    chunk_latch <= chunk_index;
                    write_pos <= 0;
                    load_busy <= 1;
                    state <= S_LOAD;
                end else if (start_bit && !start_prev
                    && (!STREAM_INPUT || chunks_loaded == 64)
                    && requested_sample < NUM_SAMPLES) begin
                    sample_id <= requested_sample;
                    if (STREAM_INPUT) chunks_loaded <= 0;
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
            S_LOAD: begin
                if (STREAM_INPUT)
                    images[{chunk_latch, write_pos}] <= payload_latch[write_pos * 8 +: 8];
                if (write_pos == 15) begin
                    chunks_loaded <= chunks_loaded + 1'b1;
                    load_busy <= 0;
                    state <= S_IDLE;
                end else write_pos <= write_pos + 1'b1;
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
                        if (kx == 2) begin
                            kx <= 0;
                            if (ky == 2) begin
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
                        if ((layer == 0 && ox == 29) || (layer == 1 && ox == 12)) begin
                            ox <= 0;
                            if ((layer == 0 && oy == 29) || (layer == 1 && oy == 12)) begin
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
                if ((layer == 0 && pool_idx == 1349) ||
                    (layer == 1 && pool_idx == 36 * CONV2_CHANNELS - 1)) begin
                    layer <= layer + 1'b1;
                    out_idx <= 0;
                    oc <= 0; ox <= 0; oy <= 0;
                    state <= S_INIT;
                end else begin
                    pool_idx <= pool_idx + 1'b1;
                    if ((layer == 0 && px == 14) || (layer == 1 && px == 5)) begin
                        px <= 0;
                        if ((layer == 0 && py == 14) || (layer == 1 && py == 5)) begin
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
