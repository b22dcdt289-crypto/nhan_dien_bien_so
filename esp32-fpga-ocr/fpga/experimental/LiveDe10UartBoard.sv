`timescale 1ns/1ps
// ESP32-S3 sends one 28x28 grayscale glyph; DE10-Lite runs LeNet,
// confusion-aware calibration, top-3 ranking, and per-glyph accept/reject.
// Request (790 bytes): A5 01 frame_id glyph_id + 784 pixels + CRC16-BE.
// Reply (25 bytes): 5A A5 frame glyph status flags raw bin support[2]
//                   top1/id+Q15, top2/id+Q15, top3/id+Q15, cycles[4], CRC16.
module UartRxFast #(parameter integer DIV=27)(
 input wire clk, input wire reset, input wire rx,
 output reg [7:0] data=0, output reg valid=0, output reg error=0);
 (* async_reg="true" *) reg meta=1'b1, sync_rx=1'b1;
 reg [2:0] state=0, bitidx=0; reg [15:0] ticks=0;
 always @(posedge clk) begin
  meta<=rx; sync_rx<=meta; valid<=0; error<=0;
  if(reset) begin state<=0; ticks<=0; end
  else case(state)
   0: if(!sync_rx) begin ticks<=DIV/2-1; state<=1; end
   1: if(ticks!=0) ticks<=ticks-1; else if(!sync_rx) begin ticks<=DIV-1; bitidx<=0; state<=2; end else state<=0;
   2: if(ticks!=0) ticks<=ticks-1; else begin data[bitidx]<=sync_rx; ticks<=DIV-1; if(bitidx==7) state<=3; else bitidx<=bitidx+1; end
   3: if(ticks!=0) ticks<=ticks-1; else begin valid<=sync_rx; error<=!sync_rx; state<=sync_rx?0:4; end
   4: if(sync_rx) state<=0;
   default: state<=0;
  endcase
 end
endmodule

module UartTxFast #(parameter integer DIV=27)(
 input wire clk,input wire reset,input wire start,input wire [7:0] data,
 output reg tx=1'b1,output reg busy=1'b0);
 reg [9:0] shift=10'h3ff; reg [3:0] bitidx=0; reg [15:0] ticks=0;
 always @(posedge clk) begin
  if(reset) begin tx<=1; busy<=0; end
  else if(!busy) begin
   tx<=1;
   if(start) begin shift<={1'b1,data,1'b0}; tx<=0; busy<=1; bitidx<=0; ticks<=DIV-1; end
  end else if(ticks!=0) ticks<=ticks-1;
  else if(bitidx==9) begin tx<=1; busy<=0; end
  else begin shift<={1'b1,shift[9:1]}; tx<=shift[1]; bitidx<=bitidx+1; ticks<=DIV-1; end
 end
endmodule

module LiveDe10UartBoard #(
 parameter integer UART_DIV=6
)(input wire MAX10_CLK1_50, input wire UART_RX, output wire UART_TX,
  output wire LEDR0, output wire LEDR1, output wire LEDR2);
 localparam integer PIXELS=784, CHUNKS=49, RESPONSE_BYTES=25;
 localparam [4:0] T_RX=0,T_CHUNK=1,T_COMPLETE=2,T_START=3,T_INFER=4,T_ABORT=5,
                  T_EXP_READ=6,T_EXP_CAPTURE=7,T_NORM_INIT=8,T_NORM_STEP=9,
                  T_NORM_CAPTURE=10,T_BIN=11,T_META_READ=12,T_META_CAPTURE=13,
                  T_POST_READ=14,T_POST_CAPTURE=15,T_DECIDE_FINISH=16,
                  T_POST_SUM=17,T_POST_SORT=18;
 localparam [1:0] TX_IDLE=0,TX_LAUNCH=1,TX_WAIT=2;
 reg [15:0] powerup=0; wire reset=!(&powerup);
 always @(posedge MAX10_CLK1_50) if(reset) powerup<=powerup+1'b1;
 wire [7:0] rx_data,phy_rx_data; wire rx_valid,rx_error,tx_busy;
 wire phy_rx_valid,phy_rx_error;
 reg tx_start=0; reg [7:0] tx_data=0;
 UartRxFast #(.DIV(UART_DIV)) receiver(MAX10_CLK1_50,reset,UART_RX,phy_rx_data,phy_rx_valid,phy_rx_error);
 UartTxFast #(.DIV(UART_DIV)) transmitter(MAX10_CLK1_50,reset,tx_start,tx_data,UART_TX,tx_busy);

 function automatic [15:0] crc_byte(input [15:0] crc_in,input [7:0] value);
  reg [15:0] c; integer k;
  begin c=crc_in^{value,8'h00}; for(k=0;k<8;k=k+1)c=c[15]?(c<<1)^16'h1021:c<<1; crc_byte=c; end
 endfunction

 // The ESP32 sends one glyph at a time. The FPGA validates its packet and
 // streams each 16-pixel word directly into the LeNet input buffer.
 reg [2:0] parser=0; reg [7:0] rx_frame=0,rx_glyph=0,rx_crc_hi=0;
 reg [9:0] pixel_index=0; reg [15:0] rx_crc=16'hffff;
 reg [127:0] rx_chunk=0; reg [5:0] rx_chunk_index=0; reg rx_chunk_ready=0;
 reg rx_frame_pulse=0,rx_frame_bad=0;
 always @(posedge MAX10_CLK1_50) begin
  rx_chunk_ready<=0; rx_frame_pulse<=0;
  if(reset) begin parser<=0; pixel_index<=0; rx_crc<=16'hffff; end
  else if(rx_error) begin parser<=0; pixel_index<=0; end
  else if(rx_valid) case(parser)
   0: if(rx_data==8'hA5) begin rx_crc<=crc_byte(16'hffff,rx_data); parser<=1; end
   1: if(rx_data==8'h01) begin rx_crc<=crc_byte(rx_crc,rx_data); parser<=2; end
      else if(rx_data==8'hA5) begin rx_crc<=crc_byte(16'hffff,rx_data); parser<=1; end
      else parser<=0;
   2: begin rx_frame<=rx_data; rx_crc<=crc_byte(rx_crc,rx_data); parser<=3; end
   3: begin rx_glyph<=rx_data; rx_crc<=crc_byte(rx_crc,rx_data); pixel_index<=0; parser<=4; end
   4: begin
      rx_chunk[pixel_index[3:0]*8 +: 8]<=rx_data; rx_crc<=crc_byte(rx_crc,rx_data);
      if(pixel_index[3:0]==15) begin rx_chunk_index<=pixel_index[9:4]; rx_chunk_ready<=1; end
      if(pixel_index==PIXELS-1) parser<=5; else pixel_index<=pixel_index+1'b1;
     end
   5: begin rx_crc_hi<=rx_data; parser<=6; end
   6: begin rx_frame_bad<=({rx_crc_hi,rx_data}!=rx_crc); rx_frame_pulse<=1; parser<=0; end
   default: parser<=0;
  endcase
 end

 reg [143:0] host_source=0; reg [6:0] host_sample=0; reg [5:0] host_chunk=0;
 reg host_toggle=0,host_write_valid=0,host_start=0;
 reg [6:0] expected_chunks=0; reg [3:0] start_hold=0;
 wire [6:0] chunks_loaded; wire load_busy,core_done; wire [4:0] core_prediction;
 wire [31:0] input_digest; wire [6:0] inference_tag; wire score_valid;
 wire [4:0] score_index; wire signed [31:0] score_data; wire core_led;
 reg signed [31:0] logits[0:29];

 // Calibration and fixed-point postprocessor ROMs. All tables were frozen
 // from the synthetic calibration set; inference receives pixels only, never a label.
 reg [15:0] exp_rom[0:256];
 reg [15:0] posterior_rom[0:5399];
 reg [31:0] meta_rom[0:179];
 initial begin
  $readmemh("generated/exp_q15.hex",exp_rom);
  $readmemh("generated/confusion_posterior_q15.hex",posterior_rom);
  $readmemh("generated/confusion_meta.hex",meta_rom);
 end
 reg [15:0] exp_buffer[0:29];
 reg [15:0] probability_q15[0:29];
 reg [20:0] exp_sum=0;
 reg [4:0] class_index=0,raw_class=0;
 reg [2:0] confidence_bin=0;
 reg [15:0] exp_rom_q=0,posterior_q=0,blend_q=0,support_q=0;
 reg [31:0] meta_rom_q=0;
 reg [15:0] top1_probability=0,top2_probability=0,top3_probability=0;
 reg [4:0] top1_class=0,top2_class=0,top3_class=0;
 reg [31:0] softmax_product_q=0,matrix_product_q=0;
 reg [15:0] adjusted_probability_q=0;
 reg [31:0] runtime_cycles=0,character_cycles=0;
 reg [7:0] response_frame=0,response_glyph=0,response_status=0,response_flags=0;
 reg [4:0] control_state=T_RX;
 // Store incoming bytes while the core is busy. A framing error travels as a
 // ninth bit so it reaches the parser in order. Overflow latches a fail-stop
 // state and LEDR2 until board reset; no unverified data is accepted after it.
 localparam integer RX_FIFO_DEPTH=2048;
 reg [8:0] rx_fifo [0:RX_FIFO_DEPTH-1];
 reg [10:0] fifo_wr=0,fifo_rd=0;
 reg [11:0] fifo_count=0;
 reg [8:0] fifo_out=0;
 reg fifo_pending=0,fifo_overflow=0;
 wire fifo_in=phy_rx_valid||phy_rx_error;
 wire fifo_pop=!reset&&!fifo_overflow&&(fifo_count!=0)&&!fifo_pending&&
               (control_state==T_RX)&&!rx_chunk_ready&&!rx_frame_pulse;
 wire fifo_push=!reset&&!fifo_overflow&&fifo_in&&(fifo_count<RX_FIFO_DEPTH);
 assign rx_data=fifo_out[7:0];
 assign rx_valid=fifo_pending&&!fifo_out[8]&&!fifo_overflow;
 assign rx_error=fifo_pending&&fifo_out[8]&&!fifo_overflow;
 always @(posedge MAX10_CLK1_50) begin
  if(reset) begin fifo_wr<=0; fifo_rd<=0; fifo_count<=0; fifo_pending<=0; fifo_overflow<=0; end
  else begin
   fifo_pending<=fifo_pop;
   if(fifo_in&&(fifo_count==RX_FIFO_DEPTH)) fifo_overflow<=1;
   if(fifo_push) begin rx_fifo[fifo_wr]<={phy_rx_error,phy_rx_data}; fifo_wr<=fifo_wr+1'b1; end
   if(fifo_pop) begin fifo_out<=rx_fifo[fifo_rd]; fifo_rd<=fifo_rd+1'b1; end
   case({fifo_push,fifo_pop})
    2'b10:fifo_count<=fifo_count+1'b1;
    2'b01:fifo_count<=fifo_count-1'b1;
    default:begin end
   endcase
  end
 end

 reg [1:0] tx_state=TX_IDLE; reg [7:0] tx_index=0; reg [15:0] tx_crc=16'hffff;
 reg tx_seen_busy=0,tx_pending=0; reg [31:0] accepted_frames=0,rejected_frames=0;
 reg [31:0] div_numerator=0,div_quotient=0;
 reg [20:0] div_denominator=1,div_remainder=0;
 reg [5:0] div_step=0;
 reg [7:0] mask_frame=0; reg mask_frame_valid=0; reg [7:0] accepted_mask=0;

 wire signed [32:0] logit_delta = $signed(logits[raw_class]) - $signed(logits[class_index]);
 wire [8:0] exp_address = (logit_delta<=0) ? 9'd0 :
                          (logit_delta>=33'sd32768) ? 9'd256 : {1'b0,logit_delta[14:7]};
 wire [8:0] meta_address = raw_class*9'd6 + confidence_bin;
 wire [12:0] posterior_address = class_index*13'd180 + raw_class*13'd6 + confidence_bin;
 wire [21:0] remainder_shift = {1'b0,div_remainder[19:0],div_numerator[31]};
 wire [21:0] denominator_extended = {1'b0,div_denominator};
 wire [31:0] division_next_quotient = remainder_shift>=denominator_extended ?
                                      {div_quotient[30:0],1'b1} : {div_quotient[30:0],1'b0};
 wire [20:0] division_next_remainder = remainder_shift>=denominator_extended ?
                                      (remainder_shift-denominator_extended) : remainder_shift[20:0];

 wire [15:0] softmax_weight_q15 = 16'd32768 - blend_q;
 wire [31:0] softmax_product = softmax_weight_q15 * probability_q15[class_index];
 wire [31:0] matrix_product = blend_q * posterior_q;
 wire [32:0] adjusted_sum_q = {1'b0,softmax_product_q} + {1'b0,matrix_product_q} + 33'd16384;
 // One-row automotive plates use DD-L-DDDDD. Keep the position prior in the
 // FPGA: glyph 2 is a letter, every other glyph is a digit. Probabilities are
 // not renormalized after masking, so weak evidence still fails acceptance.
 wire class_allowed_for_position = (response_glyph==8'd2) ?
                                    (class_index>=5'd10) : (class_index<5'd10);
 wire character_accepted = (top1_probability>=16'd23593) &&
                           ((top1_probability-top2_probability)>=16'd9831) &&
                           ((top1_class==raw_class)||(support_q>=16'd20));
 wire same_mask_frame = mask_frame_valid && (mask_frame==response_frame);
 wire [7:0] mask_before_glyph = same_mask_frame ? accepted_mask : 8'd0;
 wire plate_accepted = (response_glyph==8'd7) && (&mask_before_glyph[6:0]) && character_accepted;

 assign LEDR0=(control_state==T_INFER)||(control_state>=T_EXP_READ&&control_state<=T_DECIDE_FINISH);
 assign LEDR1=accepted_frames[0]; assign LEDR2=rejected_frames[0]|fifo_overflow;
 always @* begin
  host_source=0; host_source[143]=host_write_valid; host_source[142:136]=host_sample;
  host_source[135]=host_start; host_source[134]=host_toggle; host_source[133:128]=host_chunk;
  host_source[127:0]=rx_chunk;
 end
 OcrBenchCoreK3 #(.CONV2_CHANNELS(16),.STREAM_INPUT(1),.NUM_SAMPLES(1),
   .CORE_ENABLE_DIV2(0),.INPUT_SIZE(28),.REQUIRE_WRITE_VALID(1),
   .HOST_SOURCE(1),.GRAY_U8_INPUT(1)) core(
   .MAX10_CLK1_50(MAX10_CLK1_50),.host_source(host_source),.LEDR0(core_led),
   .logit_valid(score_valid),.logit_index(score_index),.logit_data(score_data),
   .input_digest(input_digest),.inference_done(core_done),.inference_pred(core_prediction),
   .inference_tag(inference_tag),.stream_chunks_loaded(chunks_loaded),.stream_load_busy(load_busy));

 // Receive / infer / calibrated decision pipeline.
 always @(posedge MAX10_CLK1_50) begin
  if(score_valid) logits[score_index]<=score_data;
  if(reset) begin
   control_state<=T_RX; host_write_valid<=0; host_start<=0; tx_pending<=0;
   exp_sum<=0; accepted_mask<=0; mask_frame_valid<=0;
  end else begin
   if(tx_state==TX_IDLE&&tx_pending) tx_pending<=0;
   case(control_state)
    T_RX: begin
      host_write_valid<=0; host_start<=0;
      if(rx_frame_pulse) begin
        response_frame<=rx_frame; response_glyph<=rx_glyph; start_hold<=0;
        if(rx_frame_bad) begin host_sample<=7'd127; control_state<=T_ABORT; end
        else control_state<=T_COMPLETE;
      end else if(rx_chunk_ready) begin
        host_chunk<=rx_chunk_index; host_toggle<=~host_toggle; host_write_valid<=1;
        expected_chunks<={1'b0,rx_chunk_index}+1'b1; control_state<=T_CHUNK;
      end
    end
    T_CHUNK: if(!load_busy&&chunks_loaded==expected_chunks) begin host_write_valid<=0; control_state<=T_RX; end
    T_COMPLETE: if(!load_busy&&chunks_loaded==CHUNKS) begin
      host_sample<=0; start_hold<=0; runtime_cycles<=0; control_state<=T_START;
    end
    T_START: begin
      if(start_hold==0) runtime_cycles<=0; else runtime_cycles<=runtime_cycles+1'b1;
      host_chunk<=0; host_start<=1;
      if(start_hold==7) begin host_start<=0; control_state<=T_INFER; end else start_hold<=start_hold+1'b1;
    end
    T_INFER: begin
      runtime_cycles<=runtime_cycles+1'b1;
      if(core_done) begin
        character_cycles<=runtime_cycles+1'b1; raw_class<=core_prediction; class_index<=0;
        exp_sum<=0; control_state<=T_EXP_READ;
      end
    end
    T_EXP_READ: begin exp_rom_q<=exp_rom[exp_address]; control_state<=T_EXP_CAPTURE; end
    T_EXP_CAPTURE: begin
      exp_buffer[class_index]<=exp_rom_q; exp_sum<=exp_sum+exp_rom_q;
      if(class_index==29) begin class_index<=0; control_state<=T_NORM_INIT; end
      else begin class_index<=class_index+1'b1; control_state<=T_EXP_READ; end
    end
    T_NORM_INIT: begin
      div_numerator<={1'b0,exp_buffer[class_index],15'b0};
      div_denominator<=exp_sum; div_remainder<=0; div_quotient<=0; div_step<=0;
      control_state<=T_NORM_STEP;
    end
    T_NORM_STEP: begin
      div_remainder<=division_next_remainder; div_quotient<=division_next_quotient;
      div_numerator<={div_numerator[30:0],1'b0};
      if(div_step==31) control_state<=T_NORM_CAPTURE;
      else div_step<=div_step+1'b1;
    end
    T_NORM_CAPTURE: begin
      probability_q15[class_index]<=div_quotient[15:0];
      if(class_index==29) control_state<=T_BIN;
      else begin class_index<=class_index+1'b1; control_state<=T_NORM_INIT; end
    end
    T_BIN: begin
      if(probability_q15[raw_class]>=16'd32441) confidence_bin<=5;
      else if(probability_q15[raw_class]>=16'd31130) confidence_bin<=4;
      else if(probability_q15[raw_class]>=16'd27853) confidence_bin<=3;
      else if(probability_q15[raw_class]>=16'd22938) confidence_bin<=2;
      else if(probability_q15[raw_class]>=16'd16384) confidence_bin<=1;
      else confidence_bin<=0;
      control_state<=T_META_READ;
    end
    T_META_READ: begin
      meta_rom_q<=meta_rom[meta_address]; control_state<=T_META_CAPTURE;
    end
    T_META_CAPTURE: begin
      support_q<=meta_rom_q[31:16]; blend_q<=meta_rom_q[15:0];
      class_index<=0; top1_probability<=0; top2_probability<=0; top3_probability<=0;
      top1_class<=31; top2_class<=31; top3_class<=31; control_state<=T_POST_READ;
    end
    T_POST_READ: begin posterior_q<=posterior_rom[posterior_address]; control_state<=T_POST_CAPTURE; end
    T_POST_CAPTURE: begin
      softmax_product_q<=softmax_product; matrix_product_q<=matrix_product;
      control_state<=T_POST_SUM;
    end
    T_POST_SUM: begin
      adjusted_probability_q<=adjusted_sum_q[30:15]; control_state<=T_POST_SORT;
    end
    T_POST_SORT: begin
      if(class_allowed_for_position) begin
        // Empty rank slots must also accept zero-valued Q15 probabilities.
        // This keeps three distinct valid classes when softmax underflows.
        if(top1_class==31 || adjusted_probability_q>top1_probability) begin
          top3_probability<=top2_probability; top3_class<=top2_class;
          top2_probability<=top1_probability; top2_class<=top1_class;
          top1_probability<=adjusted_probability_q; top1_class<=class_index;
        end else if(top2_class==31 || adjusted_probability_q>top2_probability) begin
          top3_probability<=top2_probability; top3_class<=top2_class;
          top2_probability<=adjusted_probability_q; top2_class<=class_index;
        end else if(top3_class==31 || adjusted_probability_q>top3_probability) begin
          top3_probability<=adjusted_probability_q; top3_class<=class_index;
        end
      end
      if(class_index==29) control_state<=T_DECIDE_FINISH;
      else begin class_index<=class_index+1'b1; control_state<=T_POST_READ; end
    end
    T_DECIDE_FINISH: begin
      response_status<=0; response_flags<={6'b0,plate_accepted,character_accepted};
      tx_pending<=1; control_state<=T_RX;
      mask_frame<=response_frame; mask_frame_valid<=1;
      if(response_glyph<8) begin
        accepted_mask<=(mask_before_glyph & ~(8'b1<<response_glyph)) |
                       (character_accepted ? (8'b1<<response_glyph) : 8'b0);
      end
      if(response_glyph==7) begin
        if(plate_accepted) accepted_frames<=accepted_frames+1'b1;
        else rejected_frames<=rejected_frames+1'b1;
      end
    end
    T_ABORT: begin
      host_write_valid<=0; host_sample<=7'd127;
      if(start_hold<8) begin host_start<=1; start_hold<=start_hold+1'b1; end
      else begin
        host_start<=0;
        if(chunks_loaded==0) begin
          response_status<=1; response_flags<=0; raw_class<=0; confidence_bin<=0; support_q<=0;
          top1_probability<=0; top2_probability<=0; top3_probability<=0;
          top1_class<=0; top2_class<=0; top3_class<=0; character_cycles<=0;
          tx_pending<=1; rejected_frames<=rejected_frames+1'b1; control_state<=T_RX;
        end
      end
    end
    default: control_state<=T_RX;
   endcase
  end
 end

 // Compact result packet. Only the FPGA-calculated decision and top-3 scores
 // cross UART; ESP32 does not apply softmax, calibration, or thresholds.
 always @(posedge MAX10_CLK1_50) begin
  tx_start<=0;
  if(reset) begin tx_state<=TX_IDLE; tx_index<=0; tx_crc<=16'hffff; tx_seen_busy<=0; end
  else case(tx_state)
   TX_IDLE: if(tx_pending) begin tx_index<=0; tx_crc<=16'hffff; tx_state<=TX_LAUNCH; end
   TX_LAUNCH: if(!tx_busy) begin
     case(tx_index)
      0:tx_data<=8'h5A; 1:tx_data<=8'hA5; 2:tx_data<=response_frame; 3:tx_data<=response_glyph;
      4:tx_data<=response_status; 5:tx_data<=response_flags; 6:tx_data<={3'b0,raw_class};
      7:tx_data<={5'b0,confidence_bin}; 8:tx_data<=support_q[7:0]; 9:tx_data<=support_q[15:8];
      10:tx_data<={3'b0,top1_class}; 11:tx_data<=top1_probability[7:0]; 12:tx_data<=top1_probability[15:8];
      13:tx_data<={3'b0,top2_class}; 14:tx_data<=top2_probability[7:0]; 15:tx_data<=top2_probability[15:8];
      16:tx_data<={3'b0,top3_class}; 17:tx_data<=top3_probability[7:0]; 18:tx_data<=top3_probability[15:8];
      19:tx_data<=character_cycles[7:0]; 20:tx_data<=character_cycles[15:8];
      21:tx_data<=character_cycles[23:16]; 22:tx_data<=character_cycles[31:24];
      23:tx_data<=tx_crc[15:8]; 24:tx_data<=tx_crc[7:0];
      default:tx_data<=0;
     endcase
     if(tx_index>=2&&tx_index<=22) begin
       case(tx_index)
        2:tx_crc<=crc_byte(16'hffff,response_frame); 3:tx_crc<=crc_byte(tx_crc,response_glyph);
        4:tx_crc<=crc_byte(tx_crc,response_status); 5:tx_crc<=crc_byte(tx_crc,response_flags);
        6:tx_crc<=crc_byte(tx_crc,{3'b0,raw_class}); 7:tx_crc<=crc_byte(tx_crc,{5'b0,confidence_bin});
        8:tx_crc<=crc_byte(tx_crc,support_q[7:0]); 9:tx_crc<=crc_byte(tx_crc,support_q[15:8]);
        10:tx_crc<=crc_byte(tx_crc,{3'b0,top1_class}); 11:tx_crc<=crc_byte(tx_crc,top1_probability[7:0]);
        12:tx_crc<=crc_byte(tx_crc,top1_probability[15:8]); 13:tx_crc<=crc_byte(tx_crc,{3'b0,top2_class});
        14:tx_crc<=crc_byte(tx_crc,top2_probability[7:0]); 15:tx_crc<=crc_byte(tx_crc,top2_probability[15:8]);
        16:tx_crc<=crc_byte(tx_crc,{3'b0,top3_class}); 17:tx_crc<=crc_byte(tx_crc,top3_probability[7:0]);
        18:tx_crc<=crc_byte(tx_crc,top3_probability[15:8]); 19:tx_crc<=crc_byte(tx_crc,character_cycles[7:0]);
        20:tx_crc<=crc_byte(tx_crc,character_cycles[15:8]); 21:tx_crc<=crc_byte(tx_crc,character_cycles[23:16]);
        22:tx_crc<=crc_byte(tx_crc,character_cycles[31:24]); default:tx_crc<=tx_crc;
       endcase
     end
     tx_start<=1; tx_seen_busy<=0; tx_state<=TX_WAIT;
   end
   TX_WAIT: if(tx_busy) tx_seen_busy<=1; else if(tx_seen_busy) begin
     tx_seen_busy<=0;
     if(tx_index==RESPONSE_BYTES-1) tx_state<=TX_IDLE;
     else begin tx_index<=tx_index+1'b1; tx_state<=TX_LAUNCH; end
   end
   default:tx_state<=TX_IDLE;
  endcase
 end
endmodule
