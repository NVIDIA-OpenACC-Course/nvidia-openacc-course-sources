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
#define MAX(X,Y) ((X>Y) ? X:Y)
#define MIN(X,Y) ((X<Y) ? X:Y)
void blur5(unsigned restrict char *imgData, unsigned restrict char *out, long w, long h, long ch, long step)
{
  long x, y;
  const int filtersize = 5;
  double filter[5][5] = 
  {
     1,  1,  1,  1,  1,
     1,  2,  2,  2,  1,
     1,  2,  3,  2,  1,
     1,  2,  2,  2,  1,
     1,  1,  1,  1,  1
  };
  // The denominator for scale should be the sum
  // of non-zero elements in the filter.
  float scale = 1.0 / 35.0;
   
#pragma acc parallel loop collapse(2) gang vector copyin(imgData[0:w * h * ch]) copyout(out[0:w * h * ch])
  for ( y = 0; y < h; y++ )
  {
    for ( x = 0; x < w; x++ )
    {
      float blue = 0.0, green = 0.0, red = 0.0;
      for ( int fy = 0; fy < filtersize; fy++ )
      {
        long iy = y - (filtersize/2) + fy;
        for ( int fx = 0; fx < filtersize; fx++ )
        {
          long ix = x - (filtersize/2) + fx;
          if ( (iy<0)  || (ix<0) || 
               (iy>=h) || (ix>=w) ) continue;
          blue  += filter[fy][fx] * (float)imgData[iy * step + ix * ch];
          green += filter[fy][fx] * (float)imgData[iy * step + ix * ch + 1];
          red   += filter[fy][fx] * (float)imgData[iy * step + ix * ch + 2];
        }
      }
      out[y * step + x * ch]      = 255 - (scale * blue);
      out[y * step + x * ch + 1 ] = 255 - (scale * green);
      out[y * step + x * ch + 2 ] = 255 - (scale * red);
    }
  }
}
void blur5_blocked(unsigned restrict char *imgData, unsigned restrict char *out, long w, long h, long ch, long step)
{
  long x, y;
  const int filtersize = 5, nblocks = 8;
  double filter[5][5] = 
  {
     1,  1,  1,  1,  1,
     1,  2,  2,  2,  1,
     1,  2,  3,  2,  1,
     1,  2,  2,  2,  1,
     1,  1,  1,  1,  1
  };
  // The denominator for scale should be the sum
  // of non-zero elements in the filter.
  float scale = 1.0 / 35.0;
   
  long blocksize = h/ nblocks;
#pragma acc data copyin(imgData[:w*h*ch],filter)copyout(out[:w*h*ch])
  for ( long blocky = 0; blocky < nblocks; blocky++)
  {
    // For data copies we need to include the ghost zones for the filter
    long starty = blocky * blocksize;
    long endy   = starty + blocksize;
#pragma acc parallel loop collapse(2) gang vector
    for ( y = starty; y < endy; y++ )
    {
      for ( x = 0; x < w; x++ )
      {
        float blue = 0.0, green = 0.0, red = 0.0;
        for ( int fy = 0; fy < filtersize; fy++ )
        {
          long iy = y - (filtersize/2) + fy;
          for ( int fx = 0; fx < filtersize; fx++ )
          {
            long ix = x - (filtersize/2) + fx;
            if ( (iy<0)  || (ix<0) || 
                (iy>=h) || (ix>=w) ) continue;
            blue  += filter[fy][fx] * (float)imgData[iy * step + ix * ch];
            green += filter[fy][fx] * (float)imgData[iy * step + ix * ch + 1];
            red   += filter[fy][fx] * (float)imgData[iy * step + ix * ch + 2];
          }
        }
        out[y * step + x * ch]      = 255 - (scale * blue);
        out[y * step + x * ch + 1 ] = 255 - (scale * green);
        out[y * step + x * ch + 2 ] = 255 - (scale * red);
      }
    }
  }
}
void blur5_update(unsigned restrict char *imgData, unsigned restrict char *out, long w, long h, long ch, long step)
{
  long x, y;
  const int filtersize = 5, nblocks = 8;
  double filter[5][5] = 
  {
     1,  1,  1,  1,  1,
     1,  2,  2,  2,  1,
     1,  2,  3,  2,  1,
     1,  2,  2,  2,  1,
     1,  1,  1,  1,  1
  };
  // The denominator for scale should be the sum
  // of non-zero elements in the filter.
  float scale = 1.0 / 35.0;
   
  long blocksize = h/ nblocks;
#pragma acc data create(imgData[w*h*ch],out[w*h*ch]) copyin(filter)
  {
  for ( long blocky = 0; blocky < nblocks; blocky++)
  {
    // For data copies we need to include the ghost zones for the filter
    long starty = MAX(0,blocky * blocksize - filtersize/2);
    long endy   = MIN(h,starty + blocksize + filtersize/2);
#pragma acc update device(imgData[starty*step:(endy-starty)*step])
    starty = blocky * blocksize;
    endy = starty + blocksize;
#pragma acc parallel loop collapse(2) gang vector
    for ( y = starty; y < endy; y++ )
    {
      for ( x = 0; x < w; x++ )
      {
        float blue = 0.0, green = 0.0, red = 0.0;
        for ( int fy = 0; fy < filtersize; fy++ )
        {
          long iy = y - (filtersize/2) + fy;
          for ( int fx = 0; fx < filtersize; fx++ )
          {
            long ix = x - (filtersize/2) + fx;
            if ( (iy<0)  || (ix<0) || 
                (iy>=h) || (ix>=w) ) continue;
            blue  += filter[fy][fx] * (float)imgData[iy * step + ix * ch];
            green += filter[fy][fx] * (float)imgData[iy * step + ix * ch + 1];
            red   += filter[fy][fx] * (float)imgData[iy * step + ix * ch + 2];
          }
        }
        out[y * step + x * ch]      = 255 - (scale * blue);
        out[y * step + x * ch + 1 ] = 255 - (scale * green);
        out[y * step + x * ch + 2 ] = 255 - (scale * red);
      }
    }
#pragma acc update self(out[starty*step:blocksize*step])
  }
  }
}
void blur5_pipelined(unsigned restrict char *imgData, unsigned restrict char *out, long w, long h, long ch, long step)
{
  long x, y;
  const int filtersize = 5, nblocks = 8;
  double filter[5][5] = 
  {
     1,  1,  1,  1,  1,
     1,  2,  2,  2,  1,
     1,  2,  3,  2,  1,
     1,  2,  2,  2,  1,
     1,  1,  1,  1,  1
  };
  // The denominator for scale should be the sum
  // of non-zero elements in the filter.
  float scale = 1.0 / 35.0;
   
  long blocksize = h/ nblocks;
#pragma acc data create(imgData[w*h*ch],out[w*h*ch])
  {
  for ( long blocky = 0; blocky < nblocks; blocky++)
  {
    // For data copies we need to include the ghost zones for the filter
    long starty = MAX(0,blocky * blocksize - filtersize/2);
    long endy   = MIN(h,starty + blocksize + filtersize/2);
#pragma acc update device(imgData[starty*step:(endy-starty)*step]) async(blocky%3+1)
    starty = blocky * blocksize;
    endy = starty + blocksize;
#pragma acc parallel loop collapse(2) gang vector  async(blocky%3+1)
    for ( y = starty; y < endy; y++ )
    {
      for ( x = 0; x < w; x++ )
      {
        float blue = 0.0, green = 0.0, red = 0.0;
        for ( int fy = 0; fy < filtersize; fy++ )
        {
          long iy = y - (filtersize/2) + fy;
          for ( int fx = 0; fx < filtersize; fx++ )
          {
            long ix = x - (filtersize/2) + fx;
            if ( (iy<0)  || (ix<0) || 
                (iy>=h) || (ix>=w) ) continue;
            blue  += filter[fy][fx] * (float)imgData[iy * step + ix * ch];
            green += filter[fy][fx] * (float)imgData[iy * step + ix * ch + 1];
            red   += filter[fy][fx] * (float)imgData[iy * step + ix * ch + 2];
          }
        }
        out[y * step + x * ch]      = 255 - (scale * blue);
        out[y * step + x * ch + 1 ] = 255 - (scale * green);
        out[y * step + x * ch + 2 ] = 255 - (scale * red);
      }
    }
#pragma acc update self(out[starty*step:blocksize*step]) async(blocky%3+1)
  }
#pragma acc wait
  }
}
#include <openacc.h>
#include <omp.h>
void blur5_pipelined_multi(unsigned restrict char *imgData, unsigned restrict char *out, long w, long h, long ch, long step)
{
  const int filtersize = 5, nblocks = 32;
  double filter[5][5] = 
  {
     1,  1,  1,  1,  1,
     1,  2,  2,  2,  1,
     1,  2,  3,  2,  1,
     1,  2,  2,  2,  1,
     1,  1,  1,  1,  1
  };
  // The denominator for scale should be the sum
  // of non-zero elements in the filter.
  float scale = 1.0 / 35.0;
   
  long blocksize = h/ nblocks;
#pragma omp parallel num_threads(acc_get_num_devices(acc_device_nvidia))
  {
    int myid = omp_get_thread_num();
    acc_set_device_num(myid,acc_device_nvidia);
    int queue = 1;
#pragma acc data create(imgData[w*h*ch],out[w*h*ch])
  {
#pragma omp for schedule(static)
  for ( long blocky = 0; blocky < nblocks; blocky++)
  {
    // For data copies we need to include the ghost zones for the filter
    long starty = MAX(0,blocky * blocksize - filtersize/2);
    long endy   = MIN(h,starty + blocksize + filtersize/2);
#pragma acc update device(imgData[starty*step:(endy-starty)*step]) async(queue)
    starty = blocky * blocksize;
    endy = starty + blocksize;
#pragma acc parallel loop collapse(2) gang vector async(queue)
    for ( long y = starty; y < endy; y++ )
    {
      for ( long x = 0; x < w; x++ )
      {
        float blue = 0.0, green = 0.0, red = 0.0;
        for ( int fy = 0; fy < filtersize; fy++ )
        {
          long iy = y - (filtersize/2) + fy;
          for ( int fx = 0; fx < filtersize; fx++ )
          {
            long ix = x - (filtersize/2) + fx;
            if ( (iy<0)  || (ix<0) || 
                (iy>=h) || (ix>=w) ) continue;
            blue  += filter[fy][fx] * (float)imgData[iy * step + ix * ch];
            green += filter[fy][fx] * (float)imgData[iy * step + ix * ch + 1];
            red   += filter[fy][fx] * (float)imgData[iy * step + ix * ch + 2];
          }
        }
        out[y * step + x * ch]      = 255 - (scale * blue);
        out[y * step + x * ch + 1 ] = 255 - (scale * green);
        out[y * step + x * ch + 2 ] = 255 - (scale * red);
      }
    }
#pragma acc update self(out[starty*step:blocksize*step]) async(queue)
    queue = (queue%3)+1;
  }
#pragma acc wait
  }
  }
}
