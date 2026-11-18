/* ESP32-S3 / OV3660 camera front end and UART-connected DE10-Lite LeNet OCR. */
#include <Arduino.h>
#include <WiFi.h>
#include <Preferences.h>
#include <mbedtls/base64.h>
#include "esp_camera.h"
#include "esp_http_server.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "web_ui.h"
#include "img_converters.h"
#define PLATE_MALLOC ps_malloc
#define PLATE_FREE free
#include "PlatePreprocess.h"
static bool sharpenEnabled=true;
static plate::Stats lastPreprocessStats;
static String lastRoiDataUri,usbApiReply;
static HardwareSerial fpgaUART(1);
static bool uartEnabled=false,uartVerified=false;
static uint32_t uartCount=0,uartMicros=0;
static const char FPGA_CLASS_NAMES[]="0123456789ABCDEFGHKLMNPSTUVXYZ";
struct OcrCharacterResult {uint8_t position,predicted,raw,second,third,confidenceBin;uint16_t p1,p2,p3,support;uint32_t cycles,uartRoundtripUs;bool accepted,plateAccepted;};
struct OcrFrameResult {bool valid,accepted;uint16_t frameId,logicElements,m9kBlocks,dspBlocks;uint32_t receivedAtMs,captureUs,decodeUs,segmentUs,uartUs,totalElapsedUs;OcrCharacterResult characters[8];};
static OcrFrameResult lastOcrResult={};
static portMUX_TYPE ocrResultMux=portMUX_INITIALIZER_UNLOCKED;
static uint32_t ocrPacketsAccepted=0,ocrPacketsRejected=0;
static SemaphoreHandle_t recognitionMutex,ocrDataMutex;
static uint8_t lastGlyphPixels[8][28*28];
static char ocrLastError[128]="Chua chay nhan dang";

// USB CDC console speed is independent from the physical FPGA UART below.
static constexpr uint32_t USB_CDC_BAUD = 115200;
static constexpr uint32_t FPGA_UART_BAUD = 1843200;
static constexpr size_t FPGA_GLYPH_PIXELS = 28 * 28;
static constexpr size_t FPGA_REQUEST_BYTES = 790;
static constexpr size_t FPGA_RESPONSE_BYTES = 25;
static constexpr int CAMERA_XCLK = 15, CAMERA_SDA = 4, CAMERA_SCL = 5;
static constexpr int CAMERA_D0 = 11, CAMERA_D1 = 9, CAMERA_D2 = 8, CAMERA_D3 = 10;
static constexpr int CAMERA_D4 = 12, CAMERA_D5 = 18, CAMERA_D6 = 17, CAMERA_D7 = 16;
static constexpr int CAMERA_VSYNC = 6, CAMERA_HREF = 7, CAMERA_PCLK = 13;
struct ROI { int x, y, w, h; };
static ROI currentROI = {1500, 3750, 7000, 2500};
static Preferences prefs;
static SemaphoreHandle_t cameraMutex, settingsMutex;
static httpd_handle_t server = nullptr;
static bool cameraOK = false;
static String cameraError, accessPointSSID, accessPointPassword;
static uint32_t frameSequence = 0;
static uint16_t lastWidth = 640, lastHeight = 480;
static uint32_t lastFrameBytes = 0;

static uint16_t crc16Update(uint16_t crc,uint8_t value) {
  crc^=static_cast<uint16_t>(value)<<8;
  for(int i=0;i<8;i++)crc=(crc&0x8000)?static_cast<uint16_t>((crc<<1)^0x1021):static_cast<uint16_t>(crc<<1);
  return crc;
}
static void drawTestStroke(uint8_t *gray,int x,int y) {
  for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++) {
    const int xx=x+dx,yy=y+dy;
    if(xx>=0&&xx<28&&yy>=0&&yy<28)gray[yy*28+xx]=0;
  }
}
static void drawTestLine(uint8_t *gray,int x0,int y0,int x1,int y1) {
  const int dx=abs(x1-x0),sx=x0<x1?1:-1,dy=-abs(y1-y0),sy=y0<y1?1:-1;
  int err=dx+dy;
  for(;;) {
    drawTestStroke(gray,x0,y0);if(x0==x1&&y0==y1)break;
    const int twice=2*err;
    if(twice>=dy){err+=dy;x0+=sx;}
    if(twice<=dx){err+=dx;y0+=sy;}
  }
}

static bool validROI(int x,int y,int w,int h) {
  return x>=0&&x<=10000&&y>=0&&y<=10000&&w>=500&&w<=10000&&h>=500&&h<=10000&&x<=10000-w&&y<=10000-h;
}
// Stable web API IDs: never cast these to the driver's version-dependent enum.
static int apiFrameSize(framesize_t size) {
  if(size==FRAMESIZE_QVGA)return 5;
  if(size==FRAMESIZE_VGA)return 8;
  if(size==FRAMESIZE_SVGA)return 9;
  return -1;
}

static String escapeJSON(const String &s) {
  String out; out.reserve(s.length()+8);
  for (size_t i=0;i<s.length();i++) {
    const char c=s[i];
    if(c=='"'||c=='\\') {out+='\\';out+=c;}
    else if(c=='\n')out+="\\n";
    else if(c=='\r')out+="\\r";
    else if(static_cast<uint8_t>(c)>=32)out+=c;
  }
  return out;
}
static String sensorName() {
  if(!cameraOK)return "unknown";
  sensor_t *s=esp_camera_sensor_get();
  if(!s)return "unknown";
  if(s->id.PID==OV3660_PID)return "OV3660";
  if(s->id.PID==OV2640_PID)return "OV2640";
  if(s->id.PID==OV5640_PID)return "OV5640";
  return String("PID-0x")+String(s->id.PID,HEX);
}
static String statusJSON(const char *connection) {
  ROI savedROI={1500,3750,7000,2500};
  if(settingsMutex&&xSemaphoreTake(settingsMutex,pdMS_TO_TICKS(100))==pdTRUE){savedROI=currentROI;xSemaphoreGive(settingsMutex);}
  sensor_t *s=cameraOK?esp_camera_sensor_get():nullptr;
  String out;out.reserve(1300);
  out="{\"device\":\"ESP32-S3 N16R8\",\"firmware\":\"platecam-lenet-uart-sharpen-v4\",\"connection\":\"";
  out+=connection;out+="\",\"camera_ok\":";out+=cameraOK?"true":"false";
  out+=",\"sensor\":\""+sensorName()+"\",\"error\":\""+escapeJSON(cameraError)+"\"";
  out+=",\"width\":"+String(lastWidth)+",\"height\":"+String(lastHeight);
  out+=",\"frame_sequence\":"+String(frameSequence)+",\"last_frame_bytes\":"+String(lastFrameBytes);
  out+=",\"psram_total\":"+String(ESP.getPsramSize())+",\"psram_free\":"+String(ESP.getFreePsram());
  out+=",\"heap_free\":"+String(ESP.getFreeHeap())+",\"uptime_ms\":"+String(millis())+",\"reset_reason\":"+String((int)esp_reset_reason())+",\"caller_stack_high_water_mark\":"+String(uxTaskGetStackHighWaterMark(nullptr));
  out+=",\"network\":{\"ssid\":\""+escapeJSON(accessPointSSID)+"\",\"password\":\""+escapeJSON(accessPointPassword)+"\",\"ip\":\""+WiFi.softAPIP().toString()+"\",\"internet\":false}";
  out+=",\"roi\":{\"x\":"+String(savedROI.x)+",\"y\":"+String(savedROI.y)+",\"w\":"+String(savedROI.w)+",\"h\":"+String(savedROI.h)+"}";
  if(s) {
    out+=",\"controls\":{\"framesize\":"+String(apiFrameSize(s->status.framesize))+",\"quality\":"+String(s->status.quality);
    out+=",\"brightness\":"+String(s->status.brightness)+",\"contrast\":"+String(s->status.contrast);
    out+=",\"hmirror\":"+String(s->status.hmirror)+",\"vflip\":"+String(s->status.vflip)+"}";
  }
  out+=String(",\"preprocessing\":{\"sharpen_enabled\":")+(sharpenEnabled?"true":"false")+"}";
  out+=",\"fpga\":{\"transport\":\"uart\",\"available\":";out+=uartVerified?"true":"false";
  out+=",\"uart_enabled\":";out+=uartEnabled?"true":"false";
  out+=",\"link_verified\":";out+=uartVerified?"true":"false";
  out+=",\"baud\":"+String(FPGA_UART_BAUD)+",\"class_count\":30";
  portENTER_CRITICAL(&ocrResultMux);const bool ocrReady=lastOcrResult.valid;portEXIT_CRITICAL(&ocrResultMux);
  out+=",\"ocr_available\":";out+=ocrReady?"true":"false";
  out+=",\"mode\":\"LeNet5_INT8_UART_FPGA_DECISION\"";
  out+=",\"ocr_performed\":";out+=ocrReady?"true":"false";
  out+=",\"ocr_packets_accepted\":"+String(ocrPacketsAccepted)+",\"ocr_packets_rejected\":"+String(ocrPacketsRejected);
  out+=",\"transactions\":"+String(uartCount);
  out+=",\"last_roundtrip_us\":"+String(uartMicros)+"}}";
  return out;
}
static void serialError(Stream &io,const char *reason) {
  io.print("@ERROR ");io.println(reason);
}
static int changeControl(const char *name,int value) {
  if(!strcmp(name,"sharpen")&&(value==0||value==1)){sharpenEnabled=value!=0;prefs.putBool("sharpen",sharpenEnabled);return 0;}
  if(!cameraOK)return -1;
  if(xSemaphoreTake(cameraMutex,pdMS_TO_TICKS(2000))!=pdTRUE)return -2;
  sensor_t *s=esp_camera_sensor_get();int result=-3;
  if(s) {
    if(!strcmp(name,"framesize")) {
      if(value==5)result=s->set_framesize(s,FRAMESIZE_QVGA);
      else if(value==8)result=s->set_framesize(s,FRAMESIZE_VGA);
      else if(value==9)result=s->set_framesize(s,FRAMESIZE_SVGA);
    }
    else if(!strcmp(name,"quality")&&value>=10&&value<=30)result=s->set_quality(s,value);
    else if(!strcmp(name,"brightness")&&value>=-2&&value<=2)result=s->set_brightness(s,value);
    else if(!strcmp(name,"contrast")&&value>=-2&&value<=2)result=s->set_contrast(s,value);
    else if(!strcmp(name,"hmirror")&&(value==0||value==1))result=s->set_hmirror(s,value);
    else if(!strcmp(name,"vflip")&&(value==0||value==1))result=s->set_vflip(s,value);
    // Discard stale buffers so a capture after a control change uses new settings.
    if(result==0)for(int i=0;i<2;i++){camera_fb_t *fb=esp_camera_fb_get();if(fb)esp_camera_fb_return(fb);}
  }
  xSemaphoreGive(cameraMutex);return result;
}
static bool saveROI(int x,int y,int w,int h) {
  if(!validROI(x,y,w,h)||!settingsMutex||xSemaphoreTake(settingsMutex,pdMS_TO_TICKS(1000))!=pdTRUE)return false;
  const ROI next={x,y,w,h};
  if(currentROI.x==x&&currentROI.y==y&&currentROI.w==w&&currentROI.h==h){xSemaphoreGive(settingsMutex);return true;}
  const bool ok=prefs.putBytes("roi",&next,sizeof(next))==sizeof(next);
  if(ok)currentROI=next;
  xSemaphoreGive(settingsMutex);return ok;
}
static camera_fb_t *takeFrame() {
  if(!cameraOK||xSemaphoreTake(cameraMutex,pdMS_TO_TICKS(2000))!=pdTRUE)return nullptr;
  camera_fb_t *fb=esp_camera_fb_get();
  if(!fb){xSemaphoreGive(cameraMutex);return nullptr;}
  if(fb->format!=PIXFORMAT_JPEG){esp_camera_fb_return(fb);xSemaphoreGive(cameraMutex);return nullptr;}
  lastWidth=fb->width;lastHeight=fb->height;lastFrameBytes=fb->len;frameSequence++;
  return fb; // Caller must return frame AND release cameraMutex.
}
static void returnFrame(camera_fb_t *fb) {esp_camera_fb_return(fb);xSemaphoreGive(cameraMutex);}

struct GrayJpegSink {uint8_t *bytes;size_t capacity,length;bool overflow;};
static size_t writeGrayJpeg(void *arg,size_t index,const void *data,size_t length){
  GrayJpegSink *sink=(GrayJpegSink*)arg;
  if(index>sink->capacity||length>sink->capacity-index){sink->overflow=true;return 0;}
  memcpy(sink->bytes+index,data,length);sink->length=max(sink->length,index+length);return length;
}
static String grayDataUri(const uint8_t *gray,int w,int h){
  GrayJpegSink sink={(uint8_t*)ps_malloc((size_t)w*h+4096),(size_t)w*h+4096,0,false};
  if(!sink.bytes)return "";
  const bool ok=fmt2jpg_cb((uint8_t*)gray,(size_t)w*h,w,h,PIXFORMAT_GRAYSCALE,85,writeGrayJpeg,&sink);
  if(!ok||sink.overflow||!sink.length){free(sink.bytes);return "";}
  const size_t cap=4*((sink.length+2)/3)+1;size_t written=0;char *encoded=(char*)ps_malloc(cap);String out;
  if(encoded&&mbedtls_base64_encode((unsigned char*)encoded,cap,&written,sink.bytes,sink.length)==0){encoded[written]=0;out="data:image/jpeg;base64,";out+=encoded;}
  free(encoded);free(sink.bytes);return out;
}
static String preprocessingJSON(const plate::Stats &s){
  return String("{\"sharpen_enabled\":")+(sharpenEnabled?"true":"false")+",\"enhanced\":"+(s.enhanced?"true":"false")+",\"contrast\":"+String(s.contrast)+",\"gain_q8\":"+String(s.gainQ8)+",\"focus_before_milli\":"+String(s.focusBefore)+",\"focus_after_milli\":"+String(s.focusAfter)+",\"threshold_attempts\":"+String(s.thresholdAttempts)+",\"normalization\":\"aspect_22_margin_1_area\"}";
}
static bool segmentPlateROI(const uint8_t *rgb,int fw,int fh,const ROI &roi,
                            uint8_t *glyphs,uint8_t &found,String &error){
  const int x0=(int)((int64_t)roi.x*fw/10000),y0=(int)((int64_t)roi.y*fh/10000);
  const int rw=max(1,(int)((int64_t)roi.w*fw/10000)),rh=max(1,(int)((int64_t)roi.h*fh/10000));
  const float scale=min(1.0f,min(480.0f/rw,160.0f/rh));
  const int w=max(1,(int)lroundf(rw*scale)),h=max(1,(int)lroundf(rh*scale));
  uint8_t *gray=(uint8_t*)ps_malloc((size_t)w*h);if(!gray){error="Khong du PSRAM de tien xu ly ROI";return false;}
  for(int y=0;y<h;y++)for(int x=0;x<w;x++){
    const int sx=min(fw-1,x0+(int)((x+0.5f)*rw/w)),sy=min(fh-1,y0+(int)((y+0.5f)*rh/h));
    const uint8_t *p=rgb+3*((size_t)sy*fw+sx);
    gray[(size_t)y*w+x]=(uint8_t)((77u*p[0]+150u*p[1]+29u*p[2]+128u)>>8);
  }
  plate::Stats stats;char reason[192]={0};const bool ok=plate::run(gray,w,h,glyphs,stats,reason,sizeof(reason),sharpenEnabled);
  const String preview=grayDataUri(gray,w,h);free(gray);found=(uint8_t)stats.found;error=reason;
  if(ocrDataMutex&&xSemaphoreTake(ocrDataMutex,pdMS_TO_TICKS(500))==pdTRUE){lastPreprocessStats=stats;lastRoiDataUri=preview;xSemaphoreGive(ocrDataMutex);}
  return ok;
}

static void setUartFailure(char *reason,size_t capacity,const char *value){
  if(reason&&capacity)snprintf(reason,capacity,"%s",value);
}
static bool readUartExact(uint8_t *out,size_t length,uint64_t deadlineUs,size_t *received){
  size_t got=0;while(got<length&&esp_timer_get_time()<deadlineUs){
    while(fpgaUART.available()&&got<length)out[got++]=(uint8_t)fpgaUART.read();
    if(got<length)delay(0);
  }if(received)*received=got;return got==length;
}
static bool classifyGlyphUART(const uint8_t *gray,uint8_t frameId,uint8_t glyphId,
                              OcrCharacterResult &result,uint32_t &elapsedUs,
                              char *failure,size_t failureCapacity){
  const uint64_t started=esp_timer_get_time();
  uartVerified=false;
  auto fail=[&](const char *why)->bool{
    setUartFailure(failure,failureCapacity,why);
    elapsedUs=(uint32_t)(esp_timer_get_time()-started);uartMicros=elapsedUs;uartCount++;
    return false;
  };
  setUartFailure(failure,failureCapacity,"none");
  if(!uartEnabled||!gray)return fail(!uartEnabled?"uart_disabled":"null_glyph");
  uint8_t req[FPGA_REQUEST_BYTES],reply[FPGA_RESPONSE_BYTES];
  req[0]=0xA5;req[1]=0x01;req[2]=frameId;req[3]=glyphId;memcpy(req+4,gray,FPGA_GLYPH_PIXELS);
  uint16_t crc=0xFFFF;for(size_t i=0;i<FPGA_REQUEST_BYTES-2;i++)crc=crc16Update(crc,req[i]);
  req[FPGA_REQUEST_BYTES-2]=(uint8_t)(crc>>8);req[FPGA_REQUEST_BYTES-1]=(uint8_t)crc;
  while(fpgaUART.available())fpgaUART.read();
  const size_t written=fpgaUART.write(req,sizeof(req));fpgaUART.flush();const uint64_t deadline=started+1500000ULL;
  if(written!=sizeof(req))return fail("request_short_write");
  uint8_t prev=0,b=0,observed[4]={0};size_t observedCount=0;bool sync=false;
  while(esp_timer_get_time()<deadline){if(fpgaUART.available()){
      b=(uint8_t)fpgaUART.read();if(observedCount<sizeof(observed))observed[observedCount++]=b;
      if(prev==0x5A&&b==0xA5){reply[0]=prev;reply[1]=b;sync=true;break;}prev=b;
    }else delay(0);}
  if(!sync){
    if(!observedCount)return fail("response_sync_timeout_no_bytes");
    char why[72];snprintf(why,sizeof(why),"response_sync_timeout_rx_%02X%02X%02X%02X",
      observed[0],observedCount>1?observed[1]:0,observedCount>2?observed[2]:0,observedCount>3?observed[3]:0);
    return fail(why);
  }
  size_t received=0;
  if(!readUartExact(reply+2,FPGA_RESPONSE_BYTES-2,deadline,&received)){
    char why[48];snprintf(why,sizeof(why),"response_short_%u_of_23",(unsigned)received);return fail(why);
  }
  if(reply[2]!=frameId)return fail("frame_id_mismatch");
  if(reply[3]!=glyphId)return fail("glyph_id_mismatch");
  if(reply[4]!=0){char why[32];snprintf(why,sizeof(why),"fpga_status_%u",reply[4]);return fail(why);}
  uint16_t expected=(uint16_t)(((uint16_t)reply[23]<<8)|reply[24]);
  if(ocrCrc16(reply+2,21)!=expected)return fail("response_crc_mismatch");
  const uint16_t rawP1=(uint16_t)(reply[11]|((uint16_t)reply[12]<<8));
  const uint16_t rawP2=(uint16_t)(reply[14]|((uint16_t)reply[15]<<8));
  const uint16_t rawP3=(uint16_t)(reply[17]|((uint16_t)reply[18]<<8));
  result.position=glyphId;result.raw=reply[6]&0x1f;result.confidenceBin=reply[7]&0x07;
  result.support=(uint16_t)(reply[8]|((uint16_t)reply[9]<<8));
  result.predicted=reply[10]&0x1f;result.p1=(uint16_t)(((uint32_t)rawP1*10000u+16384u)/32768u);
  result.second=reply[13]&0x1f;result.p2=(uint16_t)(((uint32_t)rawP2*10000u+16384u)/32768u);
  result.third=reply[16]&0x1f;result.p3=(uint16_t)(((uint32_t)rawP3*10000u+16384u)/32768u);
  result.cycles=(uint32_t)reply[19]|((uint32_t)reply[20]<<8)|((uint32_t)reply[21]<<16)|((uint32_t)reply[22]<<24);
  result.accepted=(reply[5]&0x01)!=0;result.plateAccepted=(reply[5]&0x02)!=0;
  elapsedUs=(uint32_t)(esp_timer_get_time()-started);uartMicros=elapsedUs;uartCount++;uartVerified=true;return true;
}

static esp_err_t sendJSON(httpd_req_t *req,const String &json,const char *status="200 OK") {
  if(!req){usbApiReply=json;return ESP_OK;}
  httpd_resp_set_status(req,status);httpd_resp_set_type(req,"application/json; charset=utf-8");
  httpd_resp_set_hdr(req,"Cache-Control","no-store");httpd_resp_set_hdr(req,"X-Content-Type-Options","nosniff");
  return httpd_resp_send(req,json.c_str(),json.length());
}
static esp_err_t rootHandler(httpd_req_t *req) {
  httpd_resp_set_type(req,"text/html; charset=utf-8");httpd_resp_set_hdr(req,"Content-Encoding","gzip");
  httpd_resp_set_hdr(req,"Cache-Control","no-cache");httpd_resp_set_hdr(req,"X-Content-Type-Options","nosniff");
  httpd_resp_set_hdr(req,"Referrer-Policy","no-referrer");
  return httpd_resp_send(req,reinterpret_cast<const char *>(WEB_UI_GZIP),sizeof(WEB_UI_GZIP));
}
static esp_err_t statusHandler(httpd_req_t *req) {return sendJSON(req,statusJSON("wifi"));}
static esp_err_t frameHandler(httpd_req_t *req) {
  camera_fb_t *fb=takeFrame();
  if(!fb)return sendJSON(req,"{\"error\":\"Camera khong tra ve anh. Kiem tra nguon/PSRAM/cap camera.\"}","503 Service Unavailable");
  char seq[16],width[8],height[8];snprintf(seq,sizeof(seq),"%lu",static_cast<unsigned long>(frameSequence));
  snprintf(width,sizeof(width),"%u",fb->width);snprintf(height,sizeof(height),"%u",fb->height);
  const size_t jpegLength=fb->len;uint8_t *jpeg=(uint8_t*)ps_malloc(jpegLength);
  if(!jpeg){returnFrame(fb);return sendJSON(req,"{\"error\":\"Khong du PSRAM de gui anh xem truoc.\"}","503 Service Unavailable");}
  memcpy(jpeg,fb->buf,jpegLength);returnFrame(fb);
  httpd_resp_set_type(req,"image/jpeg");httpd_resp_set_hdr(req,"Cache-Control","no-store");
  httpd_resp_set_hdr(req,"X-Frame-Sequence",seq);httpd_resp_set_hdr(req,"X-Frame-Width",width);httpd_resp_set_hdr(req,"X-Frame-Height",height);
  esp_err_t result=httpd_resp_send(req,reinterpret_cast<const char *>(jpeg),jpegLength);free(jpeg);return result;
}
static bool receiveForm(httpd_req_t *req,char *body,size_t cap) {
  if(req->content_len<=0||static_cast<size_t>(req->content_len)>=cap)return false;
  size_t got=0;
  while(got<static_cast<size_t>(req->content_len)) {
    int n=httpd_req_recv(req,body+got,req->content_len-got);
    if(n<=0)return false;got+=n;
  }
  body[got]=0;return true;
}
static bool integerParam(const char *body,const char *key,int &value) {
  char field[24];if(httpd_query_key_value(body,key,field,sizeof(field))!=ESP_OK)return false;
  char *end=nullptr;long parsed=strtol(field,&end,10);
  if(end==field||*end||parsed< -10000||parsed>10000)return false;
  value=static_cast<int>(parsed);return true;
}
static esp_err_t controlHandler(httpd_req_t *req) {
  char body[160],name[24];int value;
  if(!receiveForm(req,body,sizeof(body))||httpd_query_key_value(body,"name",name,sizeof(name))!=ESP_OK||!integerParam(body,"value",value))return sendJSON(req,"{\"error\":\"Thong so khong hop le\"}","400 Bad Request");
  if(changeControl(name,value)!=0)return sendJSON(req,"{\"error\":\"Khong doi duoc thong so camera\"}","400 Bad Request");
  return sendJSON(req,"{\"ok\":true}");
}
static esp_err_t roiHandler(httpd_req_t *req) {
  char body[160];int x,y,w,h;
  if(!receiveForm(req,body,sizeof(body))||!integerParam(body,"x",x)||!integerParam(body,"y",y)||!integerParam(body,"w",w)||!integerParam(body,"h",h)||!saveROI(x,y,w,h))return sendJSON(req,"{\"error\":\"Khung ROI khong hop le\"}","400 Bad Request");
  return sendJSON(req,"{\"ok\":true}");
}
static uint16_t ocrCrc16(const uint8_t *data,size_t len){uint16_t crc=0xffff;for(size_t i=0;i<len;i++)crc=crc16Update(crc,data[i]);return crc;}
static void invalidateOcr(const char *message){
  if(ocrDataMutex&&xSemaphoreTake(ocrDataMutex,pdMS_TO_TICKS(500))==pdTRUE){
    lastOcrResult.valid=false;
    strncpy(ocrLastError,message,sizeof(ocrLastError)-1);ocrLastError[sizeof(ocrLastError)-1]=0;xSemaphoreGive(ocrDataMutex);
  }
  portENTER_CRITICAL(&ocrResultMux);lastOcrResult.valid=false;portEXIT_CRITICAL(&ocrResultMux);
}
static esp_err_t resultHandler(httpd_req_t *req){
  const uint64_t buildStarted=esp_timer_get_time();
  if(!ocrDataMutex||xSemaphoreTake(ocrDataMutex,pdMS_TO_TICKS(500))!=pdTRUE)return sendJSON(req,"{\"available\":false,\"ocr_performed\":false,\"state\":\"busy\",\"message\":\"Dang cap nhat ket qua OCR.\",\"last_result\":null}");
  const String roiPreview=lastRoiDataUri,preprocess=preprocessingJSON(lastPreprocessStats);
  OcrFrameResult r=lastOcrResult;char message[sizeof(ocrLastError)];strncpy(message,ocrLastError,sizeof(message));message[sizeof(message)-1]=0;
  uint8_t *glyphSnapshot=nullptr;
  if(r.valid){glyphSnapshot=(uint8_t*)ps_malloc(8*FPGA_GLYPH_PIXELS);if(glyphSnapshot)memcpy(glyphSnapshot,lastGlyphPixels,8*FPGA_GLYPH_PIXELS);}
  xSemaphoreGive(ocrDataMutex);
  if(!r.valid){String pending="{\"available\":false,\"ocr_performed\":false,\"state\":\"awaiting_plate\",\"message\":\"";pending+=escapeJSON(String(message));pending+="\",\"last_result\":null}";return sendJSON(req,pending);}
  String out;out.reserve(14000);out="{\"available\":true,\"ocr_performed\":true,\"state\":\"result_received\",\"input_roi_data_uri\":\""+roiPreview+"\",\"preprocessing\":"+preprocess+",\"last_result\":{";
  out+="\"frame_id\":"+String(r.frameId)+",\"accepted\":";out+=r.accepted?"true":"false";out+=",\"plate\":\"";
  char plate[9];for(uint8_t i=0;i<8;i++)plate[i]=FPGA_CLASS_NAMES[r.characters[i].predicted];plate[8]=0;
  for(uint8_t i=0;i<8;i++){if(i==3)out+="-";else if(i==6)out+=".";out+=plate[i];}
  out+="\",\"total_elapsed_us\":"+String(r.totalElapsedUs)+",\"timings\":{\"capture_us\":"+String(r.captureUs)+",\"jpeg_decode_us\":"+String(r.decodeUs)+",\"segmentation_us\":"+String(r.segmentUs)+",\"uart_fpga_us\":"+String(r.uartUs)+"},\"resources\":{\"logic_elements\":"+String(r.logicElements)+",\"m9k_blocks\":"+String(r.m9kBlocks)+",\"dsp_blocks\":"+String(r.dspBlocks)+"},\"characters\":[";
  for(uint8_t pos=0;pos<8;pos++){
    const OcrCharacterResult *c=&r.characters[pos];
    if(pos)out+=",";
    out+="{\"position\":"+String(pos)+",\"predicted\":\"";out+=FPGA_CLASS_NAMES[c->predicted];
    out+="\",\"raw\":\"";out+=FPGA_CLASS_NAMES[c->raw];
    out+="\",\"accepted\":";out+=c->accepted?"true":"false";
    out+=",\"confidence_bin\":"+String(c->confidenceBin)+",\"confusion_support\":"+String(c->support);
    out+=",\"cycles\":"+String(c->cycles)+",\"latency_us\":"+String(c->cycles/50)+",\"uart_roundtrip_us\":"+String(c->uartRoundtripUs)+",\"top3\":[";
    const uint8_t ids[3]={c->predicted,c->second,c->third};const uint16_t probs[3]={c->p1,c->p2,c->p3};
    for(uint8_t k=0;k<3;k++){if(k)out+=",";out+="{\"character\":\"";out+=FPGA_CLASS_NAMES[ids[k]];out+="\",\"probability_pct\":";out+=String(probs[k]/100.0f,2);out+="}";}
    const String glyphImage=glyphSnapshot?grayDataUri(glyphSnapshot+(size_t)pos*FPGA_GLYPH_PIXELS,28,28):String("");
    out+="],\"glyph_data_uri\":\""+glyphImage+"\"}";
  }
  out+="]}}";free(glyphSnapshot);
  char buildTime[16];snprintf(buildTime,sizeof(buildTime),"%lu",(unsigned long)(esp_timer_get_time()-buildStarted));
  if(req)httpd_resp_set_hdr(req,"X-OCR-Response-Build-Us",buildTime);return sendJSON(req,out);
}
static esp_err_t recognizeHandler(httpd_req_t *req){
  if(!recognitionMutex||xSemaphoreTake(recognitionMutex,0)!=pdTRUE)
    return sendJSON(req,"{\"error\":\"Dang co mot luot nhan dang dang chay.\"}","409 Conflict");
  if(!cameraOK||!uartEnabled){xSemaphoreGive(recognitionMutex);return sendJSON(req,"{\"error\":\"Camera hoac UART FPGA chua san sang.\"}","503 Service Unavailable");}
  invalidateOcr("Dang chup anh va chay LeNet tren FPGA...");
  const uint64_t started=esp_timer_get_time();
  uint64_t stageStarted=started;
  camera_fb_t *fb=takeFrame();
  if(!fb){invalidateOcr("Khong lay duoc khung hinh tu camera.");xSemaphoreGive(recognitionMutex);return sendJSON(req,"{\"error\":\"Khong lay duoc khung hinh tu camera.\"}","503 Service Unavailable");}
  const uint32_t captureUs=(uint32_t)(esp_timer_get_time()-stageStarted);stageStarted=esp_timer_get_time();
  const int fw=fb->width,fh=fb->height;const size_t rgbBytes=(size_t)fw*fh*3;
  uint8_t *rgb=(uint8_t*)ps_malloc(rgbBytes);
  const bool decoded=rgb&&fmt2rgb888(fb->buf,fb->len,fb->format,rgb);
  returnFrame(fb);
  const uint32_t decodeUs=(uint32_t)(esp_timer_get_time()-stageStarted);stageStarted=esp_timer_get_time();
  if(!decoded){free(rgb);invalidateOcr("Khong giai ma duoc JPEG tu OV3660.");xSemaphoreGive(recognitionMutex);return sendJSON(req,"{\"error\":\"Khong giai ma duoc JPEG tu camera.\"}","500 Internal Server Error");}
  ROI roiCopy=currentROI;
  if(settingsMutex&&xSemaphoreTake(settingsMutex,pdMS_TO_TICKS(250))==pdTRUE){roiCopy=currentROI;xSemaphoreGive(settingsMutex);}
  uint8_t *glyphs=(uint8_t*)ps_malloc(8*FPGA_GLYPH_PIXELS);uint8_t found=0;String segmentError;
  const bool segmented=glyphs&&segmentPlateROI(rgb,fw,fh,roiCopy,glyphs,found,segmentError);
  free(rgb);
  const uint32_t segmentUs=(uint32_t)(esp_timer_get_time()-stageStarted);stageStarted=esp_timer_get_time();
  if(!segmented){const bool allocated=glyphs!=nullptr;String message=allocated?segmentError:"Khong du PSRAM de tao bo ky tu.";free(glyphs);invalidateOcr(message.c_str());xSemaphoreGive(recognitionMutex);String response="{\"error\":\""+escapeJSON(message)+"\",\"glyphs_found\":"+String(found)+",\"input_roi_data_uri\":\""+lastRoiDataUri+"\",\"preprocessing\":"+preprocessingJSON(lastPreprocessStats)+"}";return sendJSON(req,response,"422 Unprocessable Entity");}

  static uint8_t frameId=0;const uint8_t thisFrame=++frameId;
  OcrFrameResult completed={};completed.frameId=thisFrame;completed.receivedAtMs=millis();
  completed.captureUs=captureUs;completed.decodeUs=decodeUs;completed.segmentUs=segmentUs;
  completed.logicElements=6958;completed.m9kBlocks=88;completed.dspBlocks=5;completed.accepted=false;
  for(uint8_t pos=0;pos<8;pos++){
    uint32_t elapsed=0;
    char failure[72];
    if(!classifyGlyphUART(glyphs+(size_t)pos*FPGA_GLYPH_PIXELS,thisFrame,pos,completed.characters[pos],elapsed,failure,sizeof(failure))){
      const uint32_t totalElapsedUs=(uint32_t)(esp_timer_get_time()-started);
      const uint32_t uartElapsedUs=(uint32_t)(esp_timer_get_time()-stageStarted);
      uint64_t completedCycles=0;
      for(uint8_t i=0;i<pos;i++)completedCycles+=completed.characters[i].cycles;
      char message[160];snprintf(message,sizeof(message),"UART %s tai ky tu %u sau %lu ms; khong chap nhan chuoi.",failure,pos+1,(unsigned long)(elapsed/1000));
      String response="{\"error\":\""+escapeJSON(String(message))+"\",\"frame_id\":"+String(thisFrame)+",\"position\":"+String(pos);
      response+=",\"failed_character_roundtrip_us\":"+String(elapsed)+",\"total_elapsed_us\":"+String(totalElapsedUs);
      response+=",\"completed_fpga_core_us\":"+String((uint32_t)(completedCycles/50));
      response+=",\"timings\":{\"capture_us\":"+String(captureUs)+",\"jpeg_decode_us\":"+String(decodeUs)+",\"segmentation_us\":"+String(segmentUs)+",\"uart_fpga_us\":"+String(uartElapsedUs)+"},\"completed_characters\":[";
      for(uint8_t i=0;i<pos;i++){
        if(i)response+=",";
        const OcrCharacterResult &c=completed.characters[i];
        response+="{\"position\":"+String(i)+",\"predicted\":\"";response+=FPGA_CLASS_NAMES[c.predicted];
        response+="\",\"cycles\":"+String(c.cycles)+",\"core_latency_us\":"+String(c.cycles/50)+",\"uart_roundtrip_us\":"+String(c.uartRoundtripUs)+"}";
      }
      response+="],\"input_roi_data_uri\":\""+lastRoiDataUri+"\",\"preprocessing\":"+preprocessingJSON(lastPreprocessStats)+"}";
      free(glyphs);invalidateOcr(message);ocrPacketsRejected++;xSemaphoreGive(recognitionMutex);
      return sendJSON(req,response,"502 Bad Gateway");
    }
    completed.characters[pos].uartRoundtripUs=elapsed;
    if(pos==7)completed.accepted=completed.characters[pos].plateAccepted;
  }
  completed.uartUs=(uint32_t)(esp_timer_get_time()-stageStarted);
  completed.totalElapsedUs=(uint32_t)(esp_timer_get_time()-started);completed.valid=true;
  if(ocrDataMutex&&xSemaphoreTake(ocrDataMutex,pdMS_TO_TICKS(1000))==pdTRUE){
    lastOcrResult=completed;memcpy(lastGlyphPixels,glyphs,8*FPGA_GLYPH_PIXELS);ocrLastError[0]=0;xSemaphoreGive(ocrDataMutex);
  }else{
    free(glyphs);invalidateOcr("Khong ghi duoc ket qua OCR vao bo nho.");xSemaphoreGive(recognitionMutex);
    return sendJSON(req,"{\"error\":\"Khong ghi duoc ket qua OCR vao bo nho.\"}","503 Service Unavailable");
  }
  free(glyphs);portENTER_CRITICAL(&ocrResultMux);lastOcrResult.valid=true;portEXIT_CRITICAL(&ocrResultMux);
  ocrPacketsAccepted++;uartVerified=true;
  char plate[12];uint8_t p=0;for(uint8_t i=0;i<8;i++){if(i==3)plate[p++]='-';else if(i==6)plate[p++]='.';plate[p++]=FPGA_CLASS_NAMES[completed.characters[i].predicted];}plate[p]=0;
  Serial.printf("@OCR frame=%u plate=%s accepted=%u total_us=%lu",thisFrame,plate,completed.accepted?1:0,(unsigned long)completed.totalElapsedUs);
  for(uint8_t i=0;i<8;i++)Serial.printf(" c%u=%c(%.2f%%,%lu cyc)",i+1,FPGA_CLASS_NAMES[completed.characters[i].predicted],completed.characters[i].p1/100.0f,(unsigned long)completed.characters[i].cycles);
  Serial.printf(" stages_us=capture:%lu,decode:%lu,segment:%lu,uart_fpga:%lu",(unsigned long)completed.captureUs,(unsigned long)completed.decodeUs,(unsigned long)completed.segmentUs,(unsigned long)completed.uartUs);
  Serial.println();xSemaphoreGive(recognitionMutex);return resultHandler(req);
}
static void startWeb() {
  httpd_config_t config=HTTPD_DEFAULT_CONFIG();config.server_port=80;config.max_uri_handlers=8;
  config.stack_size=16384;config.max_open_sockets=4;config.lru_purge_enable=true;
  config.recv_wait_timeout=4;config.send_wait_timeout=4;
  if(httpd_start(&server,&config)!=ESP_OK){Serial.println("# HTTP server failed");return;}
  httpd_uri_t root={};root.uri="/";root.method=HTTP_GET;root.handler=rootHandler;httpd_register_uri_handler(server,&root);
  httpd_uri_t stat={};stat.uri="/api/status";stat.method=HTTP_GET;stat.handler=statusHandler;httpd_register_uri_handler(server,&stat);
  httpd_uri_t frame={};frame.uri="/api/frame";frame.method=HTTP_GET;frame.handler=frameHandler;httpd_register_uri_handler(server,&frame);
  httpd_uri_t control={};control.uri="/api/control";control.method=HTTP_POST;control.handler=controlHandler;httpd_register_uri_handler(server,&control);
  httpd_uri_t roi={};roi.uri="/api/roi";roi.method=HTTP_POST;roi.handler=roiHandler;httpd_register_uri_handler(server,&roi);
  httpd_uri_t result={};result.uri="/api/result";result.method=HTTP_GET;result.handler=resultHandler;httpd_register_uri_handler(server,&result);
  httpd_uri_t recognize={};recognize.uri="/api/recognize";recognize.method=HTTP_POST;recognize.handler=recognizeHandler;httpd_register_uri_handler(server,&recognize);
}
static void startCamera() {
  if(!cameraMutex||!settingsMutex){cameraError="Khong tao duoc mutex. Kiem tra bo nho.";return;}
  if(!psramFound()){cameraError="Khong tim thay PSRAM. Chon OPI PSRAM cho N16R8.";return;}
  camera_config_t config={};
  config.ledc_channel=LEDC_CHANNEL_0;config.ledc_timer=LEDC_TIMER_0;
  config.pin_d0=CAMERA_D0;config.pin_d1=CAMERA_D1;config.pin_d2=CAMERA_D2;config.pin_d3=CAMERA_D3;
  config.pin_d4=CAMERA_D4;config.pin_d5=CAMERA_D5;config.pin_d6=CAMERA_D6;config.pin_d7=CAMERA_D7;
  config.pin_xclk=CAMERA_XCLK;config.pin_pclk=CAMERA_PCLK;config.pin_vsync=CAMERA_VSYNC;config.pin_href=CAMERA_HREF;
  config.pin_sccb_sda=CAMERA_SDA;config.pin_sccb_scl=CAMERA_SCL;config.pin_pwdn=-1;config.pin_reset=-1;
  config.xclk_freq_hz=20000000;config.pixel_format=PIXFORMAT_JPEG;config.frame_size=FRAMESIZE_SVGA;
  config.jpeg_quality=12;config.fb_count=2;config.grab_mode=CAMERA_GRAB_LATEST;config.fb_location=CAMERA_FB_IN_PSRAM;
  esp_err_t err=esp_camera_init(&config);
  if(err!=ESP_OK){cameraError=String("Camera init error 0x")+String(err,HEX)+". Kiem tra cap FPC va nguon.";return;}
  cameraOK=true;sensor_t *s=esp_camera_sensor_get();
  if(s){if(s->id.PID==OV3660_PID){s->set_vflip(s,1);s->set_saturation(s,-1);}s->set_framesize(s,FRAMESIZE_VGA);}
}
static void processSerial(Stream &io,const String &line) {
  if(line=="STATUS") {io.print("@STATUS ");io.println(statusJSON("usb"));}
  else if(line=="RESULT"||line=="RECOGNIZE") {
    usbApiReply="";
    if(line=="RESULT")resultHandler(nullptr);else recognizeHandler(nullptr);
    io.print("@RESULT ");io.println(usbApiReply);usbApiReply="";
  }
  else if(line.startsWith("FPGA GLYPH ")) {
    unsigned frame=0,position=0,expected=0;char extra;
    if(sscanf(line.c_str(),"FPGA GLYPH %u %u %u %c",&frame,&position,&expected,&extra)!=3||frame>255||position>7||expected>65535){serialError(io,"invalid_glyph_header");return;}
    io.println("@GLYPH_READY 784");io.flush();
    uint8_t glyph[784];size_t got=0;const uint64_t deadline=esp_timer_get_time()+1500000ULL;
    while(got<sizeof(glyph)&&esp_timer_get_time()<deadline){while(io.available()&&got<sizeof(glyph))glyph[got++]=(uint8_t)io.read();if(got<sizeof(glyph))delay(0);}
    if(got!=sizeof(glyph)||ocrCrc16(glyph,sizeof(glyph))!=expected){serialError(io,"usb_glyph_length_or_crc");return;}
    if(!recognitionMutex||xSemaphoreTake(recognitionMutex,0)!=pdTRUE){serialError(io,"recognizer_busy");return;}
    OcrCharacterResult c={};uint32_t elapsed=0;char failure[72];
    const bool ok=classifyGlyphUART(glyph,(uint8_t)frame,(uint8_t)position,c,elapsed,failure,sizeof(failure));
    xSemaphoreGive(recognitionMutex);
    if(!ok){serialError(io,failure);return;}
    io.printf("@GLYPH_RESULT {\"frame_id\":%u,\"position\":%u,\"raw\":\"%c\",\"predicted\":\"%c\",\"accepted\":%s,\"plate_accepted\":%s,\"cycles\":%lu,\"roundtrip_us\":%lu,\"confidence_bin\":%u,\"support\":%u,\"top3\":[{\"character\":\"%c\",\"probability_pct\":%.2f},{\"character\":\"%c\",\"probability_pct\":%.2f},{\"character\":\"%c\",\"probability_pct\":%.2f}]}\n",frame,position,FPGA_CLASS_NAMES[c.raw],FPGA_CLASS_NAMES[c.predicted],c.accepted?"true":"false",c.plateAccepted?"true":"false",(unsigned long)c.cycles,(unsigned long)elapsed,c.confidenceBin,c.support,FPGA_CLASS_NAMES[c.predicted],c.p1/100.f,FPGA_CLASS_NAMES[c.second],c.p2/100.f,FPGA_CLASS_NAMES[c.third],c.p3/100.f);
  }
  else if(line=="UART ENABLE") {
    // Use the board's explicitly labelled UART pins. UART1 is routed via the
    // ESP32-S3 GPIO matrix: GPIO43 -> TXD0 header label, GPIO44 -> RXD0 label.
    if(!uartEnabled){fpgaUART.setRxBufferSize(10000);fpgaUART.begin(FPGA_UART_BAUD,SERIAL_8N1,44,43);uartEnabled=true;}
    io.printf("@OK UART TX43 RX44 %lu 8N1\n",(unsigned long)FPGA_UART_BAUD);
  }
  else if(line=="UART DISABLE") {if(uartEnabled)fpgaUART.end();pinMode(43,INPUT);pinMode(44,INPUT);uartEnabled=false;uartVerified=false;io.println("@OK UART disabled");}
  else if(line=="FPGA TEST") {
    uint8_t glyph[FPGA_GLYPH_PIXELS];memset(glyph,255,sizeof(glyph));
    drawTestLine(glyph,9,8,14,3);drawTestLine(glyph,14,3,14,24);drawTestLine(glyph,7,24,21,24);
    static uint8_t testFrameId=0;OcrCharacterResult c={};uint32_t elapsed=0;char failure[72];const bool ok=classifyGlyphUART(glyph,testFrameId++,0,c,elapsed,failure,sizeof(failure));
    if(ok)io.printf("@FPGA_TEST class=%c confidence=%.2f%% cycles=%lu roundtrip_us=%lu accepted=%u\n",FPGA_CLASS_NAMES[c.predicted],c.p1/100.0f,(unsigned long)c.cycles,(unsigned long)elapsed,c.accepted?1:0);
    else {char message[112];snprintf(message,sizeof(message),"fpga_uart_%s_elapsed_us_%lu",failure,(unsigned long)elapsed);serialError(io,message);}
  }
  else if(line=="FRAME2") {
    camera_fb_t *fb=takeFrame();if(!fb){serialError(io,"camera_frame_unavailable");return;}
    const size_t length=fb->len;const uint16_t width=fb->width,height=fb->height;const uint32_t sequence=frameSequence;
    uint8_t *jpeg=(uint8_t*)ps_malloc(length);if(jpeg)memcpy(jpeg,fb->buf,length);returnFrame(fb);
    if(!jpeg){serialError(io,"frame_copy_psram");return;}
    // HWCDC flush holds the TX lock while waiting. The host's NEXT message
    // already provides flow control, so no blocking flush is needed here.
    io.printf("@FRAME2 %u %u %u %lu %u\n",(unsigned)length,width,height,(unsigned long)sequence,ocrCrc16(jpeg,length));
    for(size_t at=0;at<length;){
      char ready[8]={0};size_t n=0;bool newline=false;const uint64_t deadline=esp_timer_get_time()+2000000ULL;
      while(!newline&&esp_timer_get_time()<deadline){if(io.available()){const char c=io.read();if(c=='\n')newline=true;else if(c!='\r'&&n<sizeof(ready)-1)ready[n++]=c;}else delay(0);}
      if(!newline||strcmp(ready,"NEXT")){free(jpeg);serialError(io,"frame_host_ack_timeout");return;}
      // End each host-acknowledged write with a USB short packet. A 64-byte
      // multiple may stay buffered in the Windows CDC driver until timeout.
      size_t count=min((size_t)1023,length-at);if(count%64==0)--count;
      const size_t sent=io.write(jpeg+at,count);at+=count;
      if(sent!=count){free(jpeg);serialError(io,"frame_usb_short_write");return;}
    }
    io.print("\n@END2\n");free(jpeg);
  }
  else if(line=="FRAME") {
    camera_fb_t *fb=takeFrame();if(!fb){serialError(io,"camera_frame_unavailable");return;}
    char header[80];snprintf(header,sizeof(header),"@FRAME %u %u %u %lu\n",static_cast<unsigned>(fb->len),fb->width,fb->height,static_cast<unsigned long>(frameSequence));
    io.print(header);
    const size_t expected=fb->len;
    const size_t sent=io.write(fb->buf,expected);io.print("\n@END\n");io.flush();returnFrame(fb);
    if(sent!=expected)serialError(io,"short_serial_write");
  }
  else if(line.startsWith("CONTROL ")) {
    char name[24],extra;int value;
    if(sscanf(line.c_str(),"CONTROL %23s %d %c",name,&value,&extra)==2&&changeControl(name,value)==0)io.println("@OK control");else serialError(io,"invalid_control");
  }
  else if(line.startsWith("ROI ")) {
    int x,y,w,h;char extra;
    if(sscanf(line.c_str(),"ROI %d %d %d %d %c",&x,&y,&w,&h,&extra)==4&&saveROI(x,y,w,h))io.println("@OK roi");else serialError(io,"invalid_roi");
  }
  else if(line.length())serialError(io,"unknown_command");
}
void setup() {
  Serial.setRxBufferSize(4096);
  Serial.setTxBufferSize(8192);
  Serial.setTxTimeoutMs(1000);
  Serial.begin(USB_CDC_BAUD);
  // Build with USB CDC on boot so the native USB port remains the console.
  // GPIO43/44 are reserved for the ESP32<->FPGA UART and must not also host
  // the firmware console.
  fpgaUART.setRxBufferSize(10000);
  fpgaUART.begin(FPGA_UART_BAUD,SERIAL_8N1,44,43);
  uartEnabled=true;
  delay(700);Serial.printf("# PlateCam-S3 OV3660 -> DE10-Lite UART RX44/TX43 at %lu baud\n",(unsigned long)FPGA_UART_BAUD);
  cameraMutex=xSemaphoreCreateMutex();settingsMutex=xSemaphoreCreateMutex();recognitionMutex=xSemaphoreCreateMutex();ocrDataMutex=xSemaphoreCreateMutex();prefs.begin("platecam",false);sharpenEnabled=prefs.getBool("sharpen",true);
  if(prefs.getBytesLength("roi")==sizeof(ROI)) {
    ROI saved;prefs.getBytes("roi",&saved,sizeof(saved));
    if(validROI(saved.x,saved.y,saved.w,saved.h))currentROI=saved;
  }
  accessPointPassword=prefs.getString("ap_pass","");
  if(accessPointPassword.length()<12){const char chars[]="ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz23456789";accessPointPassword="";for(int i=0;i<14;i++)accessPointPassword+=chars[esp_random()%(sizeof(chars)-1)];prefs.putString("ap_pass",accessPointPassword);}
  WiFi.mode(WIFI_AP);
  // softAPmacAddress() may return all-zero before AP creation in this core.
  // Read the factory MAC instead so each board has a stable SSID suffix.
  const uint64_t factoryMAC=ESP.getEfuseMac();char suffix[5];
  snprintf(suffix,sizeof(suffix),"%02X%02X",static_cast<unsigned>((factoryMAC>>32)&0xff),static_cast<unsigned>((factoryMAC>>40)&0xff));
  accessPointSSID=String("PlateCam-")+suffix;
  WiFi.softAPConfig(IPAddress(192,168,4,1),IPAddress(192,168,4,1),IPAddress(255,255,255,0));
  if(!WiFi.softAP(accessPointSSID.c_str(),accessPointPassword.c_str(),6,0,3))Serial.println("# WiFi AP start failed");
  WiFi.setSleep(false);startCamera();startWeb();
  Serial.println("# HTTP available at http://192.168.4.1 on the device AP");
  Serial.print("@STATUS ");Serial.println(statusJSON("usb"));
}
static void pollSerial(Stream &io,String &line,bool &overflow) {
  while(io.available()) {
    char c=io.read();
    if(c=='\n'){if(!overflow)processSerial(io,line);else serialError(io,"command_too_long");line="";overflow=false;}
    else if(c!='\r'){if(line.length()<150&&!overflow)line+=c;else overflow=true;}
  }
}
void loop() {
  static String nativeLine;static bool nativeOverflow=false;
  pollSerial(Serial,nativeLine,nativeOverflow);
  delay(2);
}
SET_LOOP_TASK_STACK_SIZE(32768);
