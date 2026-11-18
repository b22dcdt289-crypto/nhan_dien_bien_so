`timescale 1ns/1ps
// Thirty-two 8-glyph plates at 100 plates/s. Sender does not wait for replies.
// This checks buffered FPGA RTL; it does not claim ESP32 hardware can sustain it.
module tb_uart_buffered;
  localparam integer DIV=6, PLATES=32, SAMPLES=27;
  reg clk=0, rx=1; wire tx,led0,led1,led2;
  always #10 clk=~clk;
  LiveDe10UartBoard #(.UART_DIV(DIV)) dut(.MAX10_CLK1_50(clk),.UART_RX(rx),.UART_TX(tx),.LEDR0(led0),.LEDR1(led1),.LEDR2(led2));
  reg [7:0] images[0:SAMPLES*784-1], exp_raw[0:SAMPLES-1], exp_pred[0:SAMPLES-1],exp_accept[0:SAMPLES-1];
  reg [15:0] exp_q15[0:SAMPLES-1];
  reg [7:0] response[0:24],value;
  reg [15:0] crc,check_crc;
  time sent_at[0:PLATES-1], completed_at[0:PLATES-1], schedule_origin;
  integer frame,pos,glyph,k,i,got,max_fifo=0;
  reg packet_ok;
  function automatic [15:0] crc_byte(input [15:0] c0,input [7:0] b);
    reg [15:0] c;integer n;
    begin c=c0^{b,8'h00};for(n=0;n<8;n=n+1)c=c[15]?(c<<1)^16'h1021:c<<1;crc_byte=c;end
  endfunction
  task automatic send_byte(input [7:0] b);
    integer n;
    begin
      @(negedge clk);rx=0;repeat(DIV)@(negedge clk);
      for(n=0;n<8;n=n+1)begin rx=b[n];repeat(DIV)@(negedge clk);end
      rx=1;repeat(DIV)@(negedge clk);
    end
  endtask
  task automatic send_crc_byte(input [7:0] b);
    begin crc=crc_byte(crc,b);send_byte(b);end
  endtask
  task automatic send_request(input integer g,input [7:0] fid,input [7:0] p,input bit corrupt);
    integer n;
    begin
      crc=16'hffff;send_crc_byte(8'ha5);send_crc_byte(8'h01);send_crc_byte(fid);send_crc_byte(p);
      for(n=0;n<784;n=n+1)send_crc_byte(images[g*784+n]);
      send_byte(crc[15:8]);send_byte(corrupt?(crc[7:0]^8'h01):crc[7:0]);
    end
  endtask
  task automatic recv_byte(output reg [7:0] b);
    integer n;
    begin
      @(negedge tx);repeat(DIV/2)@(posedge clk);
      if(tx!==0)$fatal(1,"bad UART start");
      for(n=0;n<8;n=n+1)begin repeat(DIV)@(posedge clk);b[n]=tx;end
      repeat(DIV)@(posedge clk);if(tx!==1)$fatal(1,"bad UART stop");
    end
  endtask
  always @(posedge clk) begin
    if(dut.fifo_count>max_fifo) max_fifo=dut.fifo_count;
    if(!dut.reset && dut.fifo_overflow) $fatal(1,"receive FIFO overflow");
  end
  initial begin
    $readmemh("sim_images_u8.hex",images);$readmemh("sim_expected_raw.hex",exp_raw);
    $readmemh("sim_expected_pred.hex",exp_pred);$readmemh("sim_expected_accept.hex",exp_accept);
    $readmemh("sim_expected_q15.hex",exp_q15);
    repeat(66000)@(negedge clk);schedule_origin=$time;
    fork
      begin : sender
        for(frame=0;frame<PLATES;frame=frame+1)begin
          while($time < schedule_origin+frame*10000000)@(negedge clk);
          sent_at[frame]=$time;
          for(pos=0;pos<8;pos=pos+1)begin
            glyph=(frame%3)*8+pos;
            send_request(glyph,8'h20+frame,pos,0);
          end
        end
      end
      begin : receiver
        for(got=0;got<PLATES*8;got=got+1)begin
          for(i=0;i<25;i=i+1)begin recv_byte(value);response[i]=value;end
          check_crc=16'hffff;
          for(i=2;i<=22;i=i+1)check_crc=crc_byte(check_crc,response[i]);
          if(response[0]!==8'h5a||response[1]!==8'ha5||{response[23],response[24]}!==check_crc)
            $fatal(1,"reply sync/CRC error at reply %0d",got);
          if(response[2]!==8'h20+(got/8)||response[3]!==got%8||response[4]!==0)
            $fatal(1,"reply frame/index/status error at reply %0d frame=%0d glyph=%0d status=%0d",got,response[2],response[3],response[4]);
          k=((got/8)%3)*8+(got%8);
          if(response[6]!==exp_raw[k]||response[10]!==exp_pred[k]||response[5][0]!==exp_accept[k][0]||
             {response[12],response[11]}!==exp_q15[k])
            $fatal(1,"inference mismatch reply=%0d glyph=%0d",got,k);
          if(got%8==7)begin
            completed_at[got/8]=$time;
            $display("STREAM_PLATE_OK frame=%0d latency_ns=%0d interval_ns=%0d",got/8,
                     completed_at[got/8]-sent_at[got/8],got<8?0:completed_at[got/8]-completed_at[got/8-1]);
          end
        end
      end
    join
    for(i=1;i<PLATES;i=i+1)
      if(completed_at[i]-completed_at[i-1]>10050000)$fatal(1,"throughput slower than 100 plates/s");
    fork
      send_request(0,8'hf0,0,1);
      begin for(i=0;i<25;i=i+1)begin recv_byte(value);response[i]=value;end end
    join
    check_crc=16'hffff;for(i=2;i<=22;i=i+1)check_crc=crc_byte(check_crc,response[i]);
    if(response[0]!==8'h5a||response[1]!==8'ha5||{response[23],response[24]}!==check_crc||
       response[2]!==8'hf0||response[4]!==1||response[5]!==0)$fatal(1,"bad CRC was not rejected");
    fork
      send_request(0,8'hf1,0,0);
      begin for(i=0;i<25;i=i+1)begin recv_byte(value);response[i]=value;end end
    join
    check_crc=16'hffff;for(i=2;i<=22;i=i+1)check_crc=crc_byte(check_crc,response[i]);
    if(response[0]!==8'h5a||response[1]!==8'ha5||{response[23],response[24]}!==check_crc||
       response[2]!==8'hf1||response[4]!==0||response[6]!==exp_raw[0]||response[10]!==exp_pred[0])
       $fatal(1,"did not recover correctly after bad CRC");
    $display("BUFFERED_UART_SIM_PASS plates=%0d glyphs=%0d bad_crc=1 recovered=1 max_fifo_bytes=%0d",PLATES,PLATES*8,max_fifo);
    $finish;
  end
  initial begin #(64'd500000000) $fatal(1,"buffered UART timeout count=%0d state=%0d fifo=%0d",got,dut.control_state,dut.fifo_count);end
endmodule
