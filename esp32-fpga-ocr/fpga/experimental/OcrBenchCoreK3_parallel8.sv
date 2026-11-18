`timescale 1ns/1ps
// Experimental 8-output-lane version of the v4 28x28 LeNet K3 core.
// It preserves each output's accumulation order, the INT8 weights/Tanh LUT,
// average-pool rounding and 32-bit logits. Weights are transposed offline into
// 64-bit ROM words so that all eight lanes use the same input sample each tick.
// This is an isolated prototype: it is not the firmware currently on the kit.
module OcrBenchCoreK3 #(
    parameter integer CONV2_CHANNELS = 16,
    parameter integer STREAM_INPUT = 1,
    parameter integer NUM_SAMPLES = 1,
    parameter integer CORE_ENABLE_DIV2 = 0,
    parameter integer INPUT_SIZE = 28,
    parameter integer REQUIRE_WRITE_VALID = 1,
    parameter integer HOST_SOURCE = 1,
    parameter integer GRAY_U8_INPUT = 1
) (
    input wire MAX10_CLK1_50,
    input wire [143:0] host_source,
    output wire LEDR0,
    output wire logit_valid,
    output wire [4:0] logit_index,
    output wire signed [31:0] logit_data,
    output wire [31:0] input_digest,
    output wire inference_done,
    output wire [4:0] inference_pred,
    output wire [6:0] inference_tag,
    output wire [6:0] stream_chunks_loaded,
    output wire stream_load_busy
);
    localparam integer IMAGE_PIXELS = 784, INPUT_CHUNKS = 49;
    localparam integer C1_SIDE=26, P1_SIDE=13, C2_SIDE=11, P2_SIDE=5;
    localparam integer C1_OUTPUTS=4056, P1_OUTPUTS=1014, C2_OUTPUTS=1936, P2_OUTPUTS=400;
    localparam integer WEIGHT_WORDS=7773, BIAS_WORDS=33;
    localparam [4:0] IDLE=0, LOAD=1, INIT=2, BIAS=3, PRIME=4, MAC=5,
                     SAVE=6, ADVANCE=7, POOL_READ=8, POOL_ACC=9,
                     POOL_WRITE=10, FINISH=11;

`ifndef SYNTHESIS
    initial begin
        if (CONV2_CHANNELS!=16 || STREAM_INPUT!=1 || INPUT_SIZE!=28 ||
            REQUIRE_WRITE_VALID!=1 || HOST_SOURCE!=1 || GRAY_U8_INPUT!=1 ||
            CORE_ENABLE_DIV2!=0 || NUM_SAMPLES!=1)
            $fatal(1,"parallel8 core only supports the dense v4 UART configuration");
    end
`endif

    reg [63:0] weights_packed [0:WEIGHT_WORDS-1];
    reg [255:0] biases_packed [0:BIAS_WORDS-1];
    reg signed [7:0] tanh_lut [0:256];
    reg [4:0] shifts [0:4];
    reg signed [7:0] images [0:IMAGE_PIXELS-1];
    reg signed [7:0] act_a [0:C1_OUTPUTS-1];
    reg signed [7:0] act_b [0:P1_OUTPUTS-1];
    initial begin
        $readmemh("generated/weights_parallel8.hex", weights_packed);
        $readmemh("generated/biases_parallel8.hex", biases_packed);
        $readmemh("generated/tanh.hex", tanh_lut);
        $readmemh("generated/shifts.hex", shifts);
    end

    function automatic signed [7:0] gray_u8_to_int8(input [7:0] pixel);
        reg signed [8:0] converted;
        begin
            if (pixel < 8'd128) converted=$signed({1'b0,pixel})-9'sd127;
            else converted=$signed({1'b0,pixel})-9'sd128;
            gray_u8_to_int8=converted[7:0];
        end
    endfunction
    function automatic [8:0] lut_index(input signed [31:0] value,input [4:0] shift);
        reg signed [31:0] scaled;
        begin
            scaled=value >>> shift;
            if (scaled < -32'sd128) lut_index=9'd0;
            else if (scaled > 32'sd128) lut_index=9'd256;
            else lut_index=scaled+32'sd128;
        end
    endfunction

    reg [143:0] source_meta=0, source_sync=0;
    wire start_bit=source_sync[135];
    wire [6:0] requested_sample=source_sync[142:136];
    wire [5:0] chunk_index=source_sync[133:128];
    wire write_toggle_bit=source_sync[134];
    wire write_valid=source_sync[143];
    wire abort_stream=start_bit && requested_sample==7'd127;
    reg write_toggle_seen=0, start_prev=0;
    reg [127:0] payload_latch=0;
    reg [5:0] chunk_latch=0;
    reg [3:0] write_pos=0;
    reg [6:0] chunks_loaded=0;
    reg load_busy=0;
    wire [9:0] load_addr={chunk_latch,write_pos};
    wire [7:0] load_byte=gray_u8_to_int8(payload_latch[write_pos*8 +: 8]);

    reg [4:0] state=IDLE;
    reg [2:0] layer=0;
    reg [4:0] group_id=0, ox=0, oy=0;
    reg [9:0] issued_tap=0, consumed_tap=0;
    reg [3:0] ic=0;
    reg [1:0] kx=0, ky=0;
    reg [2:0] save_lane=0;
    reg signed [31:0] accum [0:7];
    reg signed [31:0] best_score=32'sh80000000;
    reg [4:0] pred=0;
    reg [6:0] sample_id=0;
    reg busy=0, done=0;
    reg [31:0] cycles=0, result_cycles=0, digest=0;
    reg [7:0] result_seq=0;
    reg [6:0] result_sample=0;
    reg [4:0] result_pred=0;
    reg result_valid=0;

    reg [10:0] pool_idx=0;
    reg [3:0] poc=0, px=0, py=0;
    reg [1:0] pool_tap=0;
    reg signed [10:0] pool_sum=0;

    wire [9:0] taps = layer==0 ? 10'd9 : layer==1 ? 10'd54 :
                      layer==2 ? 10'd400 : layer==3 ? 10'd120 : 10'd84;
    wire [4:0] group_count = layer==0 ? 5'd1 : layer==1 ? 5'd2 :
                             layer==2 ? 5'd15 : layer==3 ? 5'd11 : 5'd4;
    wire [7:0] output_count = layer==0 ? 8'd6 : layer==1 ? 8'd16 :
                              layer==2 ? 8'd120 : layer==3 ? 8'd84 : 8'd30;
    wire [7:0] output_id = {group_id,3'b000}+save_lane;
    wire [12:0] weight_addr = layer==0 ? issued_tap :
                              layer==1 ? 13'd9 + group_id*13'd54 + issued_tap :
                              layer==2 ? 13'd117 + group_id*13'd400 + issued_tap :
                              layer==3 ? 13'd6117 + group_id*13'd120 + issued_tap :
                                         13'd7437 + group_id*13'd84 + issued_tap;
    wire [5:0] bias_addr = layer==0 ? group_id : layer==1 ? 6'd1+group_id :
                           layer==2 ? 6'd3+group_id : layer==3 ? 6'd18+group_id : 6'd29+group_id;
    wire [9:0] image_addr=(oy+ky)*28+ox+kx;
    wire [10:0] conv2_addr=ic*169+(oy+ky)*13+ox+kx;
    wire [12:0] pool_addr=layer==0 ? poc*676+(py*2+pool_tap[1])*26+px*2+pool_tap[0] :
                                          poc*121+(py*2+pool_tap[1])*11+px*2+pool_tap[0];
    wire [12:0] a_read_addr=(state==POOL_READ || state==POOL_ACC || state==POOL_WRITE) ? pool_addr : issued_tap;
    wire [10:0] b_read_addr=layer==1 ? conv2_addr : issued_tap;
    wire [12:0] output_addr=layer==0 ? output_id*676+oy*26+ox :
                            layer==1 ? output_id*121+oy*11+ox : output_id;
    wire [8:0] selected_lut_index=lut_index(accum[save_lane],shifts[layer]);

    reg [63:0] weight_q=0;
    reg [255:0] bias_q=0;
    reg signed [7:0] image_q=0, a_q=0, b_q=0;
    wire signed [7:0] operand = layer==0 ? image_q : layer==3 ? a_q : b_q;
    integer lane;

    assign logit_valid=state==SAVE && layer==4 && output_id<30;
    assign logit_index=output_id[4:0];
    assign logit_data=accum[save_lane];
    assign input_digest=digest;
    assign inference_done=state==FINISH;
    assign inference_pred=pred;
    assign inference_tag=sample_id;
    assign stream_chunks_loaded=chunks_loaded;
    assign stream_load_busy=load_busy;
    assign LEDR0=done;

    // One-cycle ROM/RAM reads. PRIME issues tap zero; MAC consumes one prior
    // read while issuing the next, keeping each lane busy every cycle.
    always @(posedge MAX10_CLK1_50) begin
        source_meta<=host_source; source_sync<=source_meta;
        start_prev<=start_bit;
        weight_q<=weights_packed[weight_addr];
        bias_q<=biases_packed[bias_addr];
        image_q<=images[image_addr];
        a_q<=act_a[a_read_addr];
        b_q<=act_b[b_read_addr];
        if (busy) cycles<=cycles+1'b1;

        case(state)
            IDLE: begin
                if (abort_stream) begin
                    chunks_loaded<=0; load_busy<=0; write_toggle_seen<=write_toggle_bit;
                    done<=0; digest<=0;
                end else if (write_valid && write_toggle_bit!=write_toggle_seen &&
                             chunks_loaded<INPUT_CHUNKS && chunk_index==chunks_loaded[5:0]) begin
                    write_toggle_seen<=write_toggle_bit;
                    payload_latch<=source_sync[127:0]; chunk_latch<=chunk_index;
                    write_pos<=0; load_busy<=1; state<=LOAD;
                    if (chunk_index==0) digest<=0;
                end else if (start_bit && !start_prev && chunks_loaded==INPUT_CHUNKS && requested_sample==0) begin
                    sample_id<=requested_sample; chunks_loaded<=0;
                    layer<=0; group_id<=0; ox<=0; oy<=0;
                    pred<=0; best_score<=32'sh80000000; cycles<=0;
                    done<=0; busy<=1; result_valid<=0; state<=INIT;
                end
            end
            LOAD: begin
                images[load_addr]<=load_byte;
                // Match the existing digest of the signed INT8 byte bit pattern.
                digest<=digest+({24'd0,load_byte}*({22'd0,load_addr}+32'd1));
                if (write_pos==15) begin
                    chunks_loaded<=chunks_loaded+1'b1; load_busy<=0; state<=IDLE;
                end else write_pos<=write_pos+1'b1;
            end
            INIT: state<=BIAS;
            BIAS: begin
                for (lane=0; lane<8; lane=lane+1) accum[lane]<=$signed(bias_q[lane*32 +: 32]);
                issued_tap<=0; consumed_tap<=0; kx<=0; ky<=0; ic<=0; state<=PRIME;
            end
            PRIME: begin
                // The ROM/RAM read at this edge is tap zero.
                if (issued_tap+1'b1<taps) begin
                    issued_tap<=issued_tap+1'b1;
                    if (layer<2) begin
                        if (kx==2) begin kx<=0; if (ky==2) begin ky<=0; ic<=ic+1'b1; end else ky<=ky+1'b1; end
                        else kx<=kx+1'b1;
                    end
                end
                state<=MAC;
            end
            MAC: begin
                for (lane=0; lane<8; lane=lane+1)
                    accum[lane]<=accum[lane]+$signed(weight_q[lane*8 +: 8])*$signed(operand);
                if (consumed_tap+1'b1==taps) begin save_lane<=0; state<=SAVE; end
                else begin
                    consumed_tap<=consumed_tap+1'b1;
                    if (issued_tap+1'b1<taps) begin
                        issued_tap<=issued_tap+1'b1;
                        if (layer<2) begin
                            if (kx==2) begin kx<=0; if (ky==2) begin ky<=0; ic<=ic+1'b1; end else ky<=ky+1'b1; end
                            else kx<=kx+1'b1;
                        end
                    end
                end
            end
            SAVE: begin
                if (output_id<output_count) begin
                    if (layer==4) begin
                        if ($signed(accum[save_lane])>$signed(best_score)) begin
                            best_score<=accum[save_lane]; pred<=output_id[4:0];
                        end
                    end else if (layer==3) act_b[output_addr]<=tanh_lut[selected_lut_index];
                    else act_a[output_addr]<=tanh_lut[selected_lut_index];
                end
                if (save_lane==7) state<=ADVANCE;
                else save_lane<=save_lane+1'b1;
            end
            ADVANCE: begin
                if (group_id+1'b1<group_count) begin group_id<=group_id+1'b1; state<=INIT; end
                else if (layer<2) begin
                    group_id<=0;
                    if ((layer==0 && ox==C1_SIDE-1 && oy==C1_SIDE-1) ||
                        (layer==1 && ox==C2_SIDE-1 && oy==C2_SIDE-1)) begin
                        pool_idx<=0; poc<=0; px<=0; py<=0; pool_tap<=0; pool_sum<=0; state<=POOL_READ;
                    end else begin
                        if ((layer==0 && ox==C1_SIDE-1) || (layer==1 && ox==C2_SIDE-1)) begin ox<=0; oy<=oy+1'b1; end
                        else ox<=ox+1'b1;
                        state<=INIT;
                    end
                end else if (layer==4) state<=FINISH;
                else begin layer<=layer+1'b1; group_id<=0; state<=INIT; end
            end
            POOL_READ: state<=POOL_ACC;
            POOL_ACC: begin
                pool_sum<=pool_sum+$signed(a_q);
                if (pool_tap==3) state<=POOL_WRITE;
                else begin pool_tap<=pool_tap+1'b1; state<=POOL_READ; end
            end
            POOL_WRITE: begin
                act_b[pool_idx]<=(pool_sum+11'sd2) >>> 2;
                if ((layer==0 && pool_idx==P1_OUTPUTS-1) || (layer==1 && pool_idx==P2_OUTPUTS-1)) begin
                    layer<=layer+1'b1; group_id<=0; ox<=0; oy<=0; state<=INIT;
                end else begin
                    pool_idx<=pool_idx+1'b1;
                    if ((layer==0 && px==P1_SIDE-1) || (layer==1 && px==P2_SIDE-1)) begin
                        px<=0;
                        if ((layer==0 && py==P1_SIDE-1) || (layer==1 && py==P2_SIDE-1)) begin py<=0; poc<=poc+1'b1; end
                        else py<=py+1'b1;
                    end else px<=px+1'b1;
                    pool_tap<=0; pool_sum<=0; state<=POOL_READ;
                end
            end
            FINISH: begin
                busy<=0; done<=1; result_cycles<=cycles+1'b1;
                result_sample<=sample_id; result_pred<=pred; result_seq<=result_seq+1'b1;
                result_valid<=1; state<=IDLE;
            end
            default: state<=IDLE;
        endcase
    end
endmodule
