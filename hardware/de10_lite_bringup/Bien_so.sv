module Bien_so (
    input  wire MAX10_CLK1_50,
    output wire LEDR0
);
    // FPGA bring-up heartbeat. This validates clock, pinout, compilation and JTAG programming.
    reg [25:0] counter = 26'd0;

    always @(posedge MAX10_CLK1_50) begin
        counter <= counter + 26'd1;
    end

    assign LEDR0 = counter[25];
endmodule
