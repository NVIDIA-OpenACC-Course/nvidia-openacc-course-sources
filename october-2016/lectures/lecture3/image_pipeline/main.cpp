/* Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <stdio.h>
#include <unistd.h>
#include <strings.h>
#include <opencv/cv.h>
#include <opencv/cvaux.h>
#include <opencv/highgui.h>
#include <omp.h>

extern "C" {
void blur5(unsigned char*,unsigned char*,long,long,long,long);
void blur5_blocked(unsigned char*,unsigned char*,long,long,long,long);
void blur5_update(unsigned char*,unsigned char*,long,long,long,long);
void blur5_pipelined(unsigned char*,unsigned char*,long,long,long,long);
void blur5_pipelined_multi(unsigned char*,unsigned char*,long,long,long,long);
}

int main(int argc, char* argv[])
{
  if (argc < 3)
  {
    fprintf(stderr,"Usage: %s inFilename outFilename\n",argv[0]);
    return -1;
  }

  IplImage* img = cvLoadImage(argv[1]);

  printf("%s: %d x %d, %d %d\n", argv[1],img->width, img->height, img->widthStep, img->nChannels);

  unsigned char *data = (unsigned char*)img->imageData;
  long width = img->width,
       height = img->height,
       ch = img->nChannels,
       ws = img->widthStep,
       sz = height * ch * ws;
  unsigned char *out = (unsigned char*)malloc(sz * sizeof(unsigned char));

  // Pre-allocate device and queues for timing
  blur5(data,out,width,height, ch, ws);
  blur5_pipelined(data,out,width,height, ch, ws);
  blur5_pipelined_multi(data,out,width,height, ch, ws);
  bzero(out,sz);

  double st = omp_get_wtime();
  blur5(data,out,width,height, ch, ws);
  double et = omp_get_wtime();
  bzero(out,sz);
  printf("Time (original): %lf seconds\n", (et-st));

  st = omp_get_wtime();
  blur5_blocked(data,out,width,height, ch, ws);
  et = omp_get_wtime();
  bzero(out,sz);
  printf("Time (blocked): %lf seconds\n", (et-st));

  st = omp_get_wtime();
  blur5_update(data,out,width,height, ch, ws);
  et = omp_get_wtime();
  bzero(out,sz);
  printf("Time (update): %lf seconds\n", (et-st));

  st = omp_get_wtime();
  blur5_pipelined(data,out,width,height, ch, ws);
  et = omp_get_wtime();
  printf("Time (pipelined): %lf seconds\n", (et-st));

  st = omp_get_wtime();
  blur5_pipelined_multi(data,out,width,height, ch, ws);
  et = omp_get_wtime();
  printf("Time (multi): %lf seconds\n", (et-st));
  memcpy(img->imageData,out,width*height*ch);

  if(!cvSaveImage(argv[2],img))
    fprintf(stderr,"Failed to write to %s.\n",argv[2]);

  cvReleaseImage(&img);

  return 0;
}
