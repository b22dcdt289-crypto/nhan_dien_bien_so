// Dedicated UART1 FPGA RAM readback test; USB only controls/displays results.
// No local echo and no OCR: received[] is filled exclusively from UART1 RX.
static size_t debugUartBytes=0;
static bool readUARTExact(uint8_t *dst,size_t n,uint32_t deadline) {
 size_t got=0;
 while(got<n&&static_cast<int32_t>(millis()-deadline)<0){
  while(fpgaUART.available()&&got<n){dst[got++]=fpgaUART.read();debugUartBytes++;}
  if(got<n)delay(1);
 }
 return got==n;
}
static void uartImageTest(Stream &io,bool camera,bool corrupt) {
 if(!uartEnabled){serialError(io,"uart_not_enabled");return;}
 const int w=camera?128:28,h=camera?64:28,n=w*h;
 uint8_t *source=(uint8_t*)ps_malloc(n),*received=(uint8_t*)ps_malloc(n);
 if(!source||!received){free(source);free(received);serialError(io,"allocation_failed");return;}
 uint32_t captureSeq=0;
 if(camera){
  camera_fb_t *fb=takeFrame();
  if(!fb){free(source);free(received);serialError(io,"camera_unavailable");return;}
  uint8_t *rgb=(uint8_t*)ps_malloc((size_t)fb->width*fb->height*3);
  bool ok=rgb&&fmt2rgb888(fb->buf,fb->len,fb->format,rgb);
  if(ok){
   ROI roi; xSemaphoreTake(settingsMutex,portMAX_DELAY);roi=currentROI;xSemaphoreGive(settingsMutex);
   const int x0=(long)roi.x*fb->width/10000,y0=(long)roi.y*fb->height/10000;
   const int rw=max(1,(int)((long)roi.w*fb->width/10000)),rh=max(1,(int)((long)roi.h*fb->height/10000));
   // Fixed ROI resize for transport inspection, not character segmentation.
   // fmt2rgb888 produces B,G,R byte order in this camera driver.
   for(int y=0;y<h;y++)for(int x=0;x<w;x++){
    int sx=min((int)fb->width-1,x0+(2*x+1)*rw/(2*w));
    int sy=min((int)fb->height-1,y0+(2*y+1)*rh/(2*h));
    const uint8_t *p=rgb+3*((size_t)sy*fb->width+sx);
    source[y*w+x]=(29*p[0]+150*p[1]+77*p[2]+128)>>8;
   }
   captureSeq=frameSequence;
  }
  free(rgb);returnFrame(fb);
  if(!ok){free(source);free(received);serialError(io,"jpeg_decode_failed");return;}
 }else{
  memset(source,255,n);drawTestLine(source,9,8,14,3);drawTestLine(source,14,3,14,24);drawTestLine(source,7,24,21,24);
 }
 static uint8_t seq=0;seq++;
 uint8_t header[7]={0xa5,0x5a,seq,(uint8_t)w,(uint8_t)h,(uint8_t)n,(uint8_t)(n>>8)};
 uint16_t crc=0xffff;for(int i=2;i<7;i++)crc=crc16Update(crc,header[i]);
 for(int i=0;i<n;i++)crc=crc16Update(crc,source[i]);
 uint8_t tail[2]={(uint8_t)(crc>>8),(uint8_t)(crc^(corrupt?1:0))};
 while(fpgaUART.available())fpgaUART.read();
 uartVerified=false;const uint32_t start=micros();
 fpgaUART.write(header,7);fpgaUART.write(source,n);fpgaUART.write(tail,2);fpgaUART.flush();
 debugUartBytes=0;uint8_t response[7]={0},crcReply[2]={0};const uint32_t deadline=millis()+(corrupt?300:2000);
 bool ok=readUARTExact(response,7,deadline);
 if(ok)ok=response[0]==0x5a&&response[1]==0xa5&&!memcmp(response+2,header+2,5);
 if(ok)ok=readUARTExact(received,n,deadline)&&readUARTExact(crcReply,2,deadline);
 uartMicros=micros()-start;
 uint16_t check=0xffff;
 if(ok){for(int i=2;i<7;i++)check=crc16Update(check,response[i]);for(int i=0;i<n;i++)check=crc16Update(check,received[i]);ok=check==((crcReply[0]<<8)|crcReply[1]);}
 if(corrupt){io.print("@BADCRC ");io.println(ok?"UNEXPECTED_REPLY":"NO_VALID_REPLY");}
 else if(!ok){
  io.printf("@ERROR uart_readback_failed received=%u header=%02x%02x%02x%02x%02x%02x%02x elapsed_us=%lu\n",
    (unsigned)debugUartBytes,response[0],response[1],response[2],response[3],response[4],response[5],response[6],(unsigned long)uartMicros);
 }
 else {
  int mismatch=0;for(int i=0;i<n;i++)if(source[i]!=received[i])mismatch++;
  uartVerified=mismatch==0;uartCount++;
  io.printf("@UARTIMAGE %d %d %d %u %lu %lu %d\n",w,h,n,seq,(unsigned long)captureSeq,(unsigned long)uartMicros,mismatch);
  io.write(source,n);io.write(received,n);io.print("\n@END\n");io.flush();
 }
 free(source);free(received);
}
