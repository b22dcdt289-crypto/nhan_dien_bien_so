// The board still receives 50 MHz at P11. The core changes state only on
// alternate rising edges, so the effective core-step rate is 25 MHz.
module Bench_dense_stream_25(
    input wire MAX10_CLK1_50,
    output wire LEDR0
);
    OcrBenchCoreK3 #(
        .CONV2_CHANNELS(16), .STREAM_INPUT(1), .NUM_SAMPLES(127),
        .CORE_ENABLE_DIV2(1)
    ) core (
        .MAX10_CLK1_50(MAX10_CLK1_50), .LEDR0(LEDR0)
    );
endmodule

module Bench_pruned_stream_25(
    input wire MAX10_CLK1_50,
    output wire LEDR0
);
    OcrBenchCoreK3 #(
        .CONV2_CHANNELS(8), .STREAM_INPUT(1), .NUM_SAMPLES(127),
        .CORE_ENABLE_DIV2(1)
    ) core (
        .MAX10_CLK1_50(MAX10_CLK1_50), .LEDR0(LEDR0)
    );
endmodule
