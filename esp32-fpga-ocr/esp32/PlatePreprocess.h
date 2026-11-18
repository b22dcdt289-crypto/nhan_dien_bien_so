#pragma once
// Shared by the ESP32 firmware and the native evaluation harness.
// Runtime inputs are pixels only; no labels or generated annotation boxes.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#ifndef PLATE_MALLOC
#define PLATE_MALLOC malloc
#endif
#ifndef PLATE_FREE
#define PLATE_FREE free
#endif
namespace plate {
template<class T> static inline T lo(T a,T b){return a<b?a:b;}
template<class T> static inline T hi(T a,T b){return a>b?a:b;}
static inline int clip(int v,int a,int b){return lo(b,hi(a,v));}
struct Box {int x0,y0,x1,y1,area;};
struct Stats {int found=0,threshold=0,contrast=0,gainQ8=256,focusBefore=0,focusAfter=0,thresholdAttempts=0;bool enhanced=false;Box boxes[8];};
static inline int focus(const uint8_t *g,int w,int h){
  uint64_t sum=0;for(int y=1;y<h-1;y++)for(int x=1;x<w-1;x++){
    const int i=y*w+x;sum+=(uint64_t)abs(4*(int)g[i]-g[i-1]-g[i+1]-g[i-w]-g[i+w]);
  }return (int)(sum*1000/(uint32_t)((w-2)*(h-2)));
}
static inline int otsu(const uint8_t *g,int n){
  uint32_t hist[256]={0};uint64_t totalSum=0,lowSum=0;uint32_t lowCount=0;
  for(int i=0;i<n;i++){hist[g[i]]++;totalSum+=g[i];}
  double best=-1;int threshold=0;
  for(int t=0;t<255;t++){lowCount+=hist[t];lowSum+=(uint64_t)t*hist[t];
    if(!lowCount||lowCount==(uint32_t)n)continue;
    const double a=(double)lowSum/lowCount,b=(double)(totalSum-lowSum)/(n-lowCount);
    const double score=(double)lowCount*(n-lowCount)*(a-b)*(a-b);
    if(score>best){best=score;threshold=t;}
  }return threshold;
}
static inline bool enhance(uint8_t *g,int w,int h,Stats &s,bool enabled){
  const int n=w*h;uint32_t hist[256]={0};for(int i=0;i<n;i++)hist[g[i]]++;
  uint32_t c=0;int p05=0,p95=255;bool first=false;
  for(int i=0;i<256;i++){c+=hist[i];if(!first&&c>=(uint32_t)hi(1,n/20)){p05=i;first=true;}if(c>=(uint32_t)(n-n/20)){p95=i;break;}}
  s.contrast=p95-p05;s.focusBefore=focus(g,w,h);s.focusAfter=s.focusBefore;
  if(s.contrast<25)return false;
  if(!enabled)return true;
  uint8_t *src=(uint8_t*)PLATE_MALLOC(n);if(!src)return false;
  // Contrast gain is capped at 1.5x; dark/light percentiles are image-derived.
  s.gainQ8=clip(215*256/hi(1,s.contrast),256,384);
  for(int i=0;i<n;i++)src[i]=(uint8_t)clip(((int(g[i])-p05)*s.gainQ8+128)/256+16,0,255);
  memcpy(g,src,n);
  for(int y=1;y<h-1;y++)for(int x=1;x<w-1;x++){
    const int i=y*w+x;
    const int blurred=(src[i-w-1]+2*src[i-w]+src[i-w+1]+2*src[i-1]+4*src[i]+2*src[i+1]+src[i+w-1]+2*src[i+w]+src[i+w+1]+8)/16;
    const int detail=int(src[i])-blurred;
    if(abs(detail)<=3)continue;
    const int correction=clip(detail*3/4,-18,18);
    int minimum=src[i],maximum=src[i];
    for(int yy=-1;yy<=1;yy++)for(int xx=-1;xx<=1;xx++){minimum=lo(minimum,(int)src[i+yy*w+xx]);maximum=hi(maximum,(int)src[i+yy*w+xx]);}
    g[i]=(uint8_t)clip(int(src[i])+correction,minimum,maximum);
  }
  PLATE_FREE(src);s.enhanced=true;s.focusAfter=focus(g,w,h);return true;
}
static inline int components(const uint8_t *g,int w,int h,int threshold,uint8_t *visited,uint32_t *queue,Box *boxes){
  memset(visited,0,w*h);int count=0;const int mx=hi(1,w/150),my=hi(1,h/50);
  for(int sy=my;sy<h-my;sy++)for(int sx=mx;sx<w-mx;sx++){
    const int seed=sy*w+sx;if(visited[seed]||g[seed]>threshold)continue;
    int head=0,tail=0;queue[tail++]=seed;visited[seed]=1;Box b={sx,sy,sx,sy,0};
    while(head<tail){const int at=(int)queue[head++],x=at%w,y=at/w;b.area++;
      b.x0=lo(b.x0,x);b.x1=hi(b.x1,x);b.y0=lo(b.y0,y);b.y1=hi(b.y1,y);
      for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
        if(!dx&&!dy)continue;const int nx=x+dx,ny=y+dy;
        if(nx<mx||nx>=w-mx||ny<my||ny>=h-my)continue;
        const int ni=ny*w+nx;if(!visited[ni]&&g[ni]<=threshold){visited[ni]=1;queue[tail++]=ni;}
      }
    }
    const int bw=b.x1-b.x0+1,bh=b.y1-b.y0+1;
    if(count<64&&b.area>=hi(6,w*h/2200)&&bh>=h*18/100&&bh<=h*94/100&&bw>=2&&bw<=w*17/100)boxes[count++]=b;
  }
  bool changed=true;while(changed){changed=false;
    for(int i=0;i<count&&!changed;i++)for(int j=i+1;j<count;j++){
      const int overlap=hi(0,lo(boxes[i].x1,boxes[j].x1)-hi(boxes[i].x0,boxes[j].x0)+1);
      const int gapY=hi(0,hi(boxes[i].y0,boxes[j].y0)-lo(boxes[i].y1,boxes[j].y1)-1);
      const int gapX=hi(0,hi(boxes[i].x0,boxes[j].x0)-lo(boxes[i].x1,boxes[j].x1)-1);
      const int overlapY=hi(0,lo(boxes[i].y1,boxes[j].y1)-hi(boxes[i].y0,boxes[j].y0)+1);
      const int minW=lo(boxes[i].x1-boxes[i].x0+1,boxes[j].x1-boxes[j].x0+1);
      const int unionW=hi(boxes[i].x1,boxes[j].x1)-lo(boxes[i].x0,boxes[j].x0)+1;
      if(unionW<=w*15/100&&((overlap*4>=minW&&gapY<=hi(2,h/20))||(gapX<=1&&overlapY>=h/3))){
        boxes[i].x0=lo(boxes[i].x0,boxes[j].x0);boxes[i].x1=hi(boxes[i].x1,boxes[j].x1);
        boxes[i].y0=lo(boxes[i].y0,boxes[j].y0);boxes[i].y1=hi(boxes[i].y1,boxes[j].y1);boxes[i].area+=boxes[j].area;
        for(int k=j;k<count-1;k++)boxes[k]=boxes[k+1];count--;changed=true;break;
      }
    }
  }
  int kept=0;for(int i=0;i<count;i++){
    const int bh=boxes[i].y1-boxes[i].y0+1;
    if(bh>=h*35/100&&boxes[i].area>=hi(8,w*h/1600))boxes[kept++]=boxes[i];
  }
  for(int i=0;i<kept;i++)for(int j=i+1;j<kept;j++)if(boxes[j].x0<boxes[i].x0){Box tmp=boxes[i];boxes[i]=boxes[j];boxes[j]=tmp;}
  return kept;
}
static inline void normalize(const uint8_t *g,int w,int h,Box b,uint8_t *dst){
  // Match training: one source-pixel margin, 22x22 aspect-preserving fit,
  // centered on a white 28x28 canvas, area resampling instead of point samples.
  b.x0=hi(0,b.x0-1);b.y0=hi(0,b.y0-1);b.x1=lo(w-1,b.x1+1);b.y1=lo(h-1,b.y1+1);
  const int bw=b.x1-b.x0+1,bh=b.y1-b.y0+1;
  const float scale=lo(22.0f/bw,22.0f/bh);const int nw=hi(1,(int)lroundf(bw*scale)),nh=hi(1,(int)lroundf(bh*scale));
  const int ox=(28-nw)/2,oy=(28-nh)/2;memset(dst,255,784);
  for(int y=0;y<nh;y++)for(int x=0;x<nw;x++){
    const int x0=x*bw*256/nw,x1=(x+1)*bw*256/nw,y0=y*bh*256/nh,y1=(y+1)*bh*256/nh;
    uint64_t sum=0,weight=0;
    for(int sy=y0/256;sy<(y1+255)/256;sy++)for(int sx=x0/256;sx<(x1+255)/256;sx++){
      const int wx=hi(0,lo(x1,(sx+1)*256)-hi(x0,sx*256));
      const int wy=hi(0,lo(y1,(sy+1)*256)-hi(y0,sy*256));const uint32_t a=(uint32_t)wx*wy;
      sum+=(uint64_t)g[(b.y0+lo(sy,bh-1))*w+b.x0+lo(sx,bw-1)]*a;weight+=a;
    }dst[(oy+y)*28+ox+x]=(uint8_t)((sum+weight/2)/hi((uint64_t)1,weight));
  }
}
static inline bool run(uint8_t *gray,int w,int h,uint8_t *glyphs,Stats &s,char *error,size_t cap,bool sharpening=true){
  s=Stats{};
  if(w<80||h<20||w>480||h>160||w<3*h||w>7*h){snprintf(error,cap,"ROI can bien oto mot hang, rong/cao 3-7");return false;}
  if(!enhance(gray,w,h,s,sharpening)){snprintf(error,cap,"ROI thieu tuong phan hoac khong du bo nho");return false;}
  const int n=w*h;uint8_t *visited=(uint8_t*)PLATE_MALLOC(n);uint32_t *queue=(uint32_t*)PLATE_MALLOC((size_t)n*sizeof(uint32_t));
  if(!visited||!queue){PLATE_FREE(visited);PLATE_FREE(queue);snprintf(error,cap,"Khong du bo nho tach ky tu");return false;}
  const int initial=otsu(gray,n),offsets[5]={0,-8,8,-16,16};Box candidates[64],best[8];int bestScore=0x7fffffff;
  for(int attempt=0;attempt<5;attempt++){
    const int t=clip(initial+offsets[attempt],20,235);s.thresholdAttempts++;
    const int count=components(gray,w,h,t,visited,queue,candidates);if(!attempt)s.found=count;
    if(count!=8)continue;
    int minCenter=10000,maxCenter=0,minHeight=h,maxHeight=0,score=abs(offsets[attempt]);
    for(int i=0;i<8;i++){const int center=candidates[i].y0+candidates[i].y1,height=candidates[i].y1-candidates[i].y0+1;
      minCenter=lo(minCenter,center);maxCenter=hi(maxCenter,center);minHeight=lo(minHeight,height);maxHeight=hi(maxHeight,height);
    }
    if(maxCenter-minCenter>h*30/100||minHeight*10<maxHeight*6||candidates[7].x1-candidates[0].x0<w/2)continue;
    score+=(maxCenter-minCenter)*8+(maxHeight-minHeight)*4;
    if(score<bestScore){bestScore=score;s.threshold=t;memcpy(best,candidates,sizeof(best));}
    if(attempt==0&&maxCenter-minCenter<h/10&&maxHeight-minHeight<h/10)break;
  }
  PLATE_FREE(visited);PLATE_FREE(queue);
  if(bestScore==0x7fffffff){snprintf(error,cap,"Tach duoc %d/8 ky tu; can bien nam ngang, tron khung va ro net",s.found);return false;}
  s.found=8;memcpy(s.boxes,best,sizeof(best));for(int i=0;i<8;i++)normalize(gray,w,h,best[i],glyphs+i*784);return true;
}
}
