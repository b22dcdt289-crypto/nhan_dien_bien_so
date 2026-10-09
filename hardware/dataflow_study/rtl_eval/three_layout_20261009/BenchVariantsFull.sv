module Bench_dense_stream(
    input wire MAX10_CLK1_50,
    output wire LEDR0
);
    OcrBenchCore #(.CONV2_CHANNELS(16), .STREAM_INPUT(1), .NUM_SAMPLES(127)) core (
        .MAX10_CLK1_50(MAX10_CLK1_50), .LEDR0(LEDR0)
    );
endmodule

module Bench_pruned_stream(
    input wire MAX10_CLK1_50,
    output wire LEDR0
);
    OcrBenchCore #(.CONV2_CHANNELS(8), .STREAM_INPUT(1), .NUM_SAMPLES(127)) core (
        .MAX10_CLK1_50(MAX10_CLK1_50), .LEDR0(LEDR0)
    );
endmodule
