#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <time.h>
#include <openacc.h>
#include <omp.h>
#include "mandelbrot.h"
using namespace std;

float myclock() {
    timespec t;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    return (float)t.tv_sec+t.tv_nsec/(float)1e9; 
}

int main() {
  float start, end;

  size_t bytes=WIDTH*HEIGHT*sizeof(unsigned char);
  unsigned char *image=(unsigned char*)malloc(bytes);
  FILE *fp=fopen("image.pgm","wb");
  fprintf(fp,"P5\n%s\n%d %d\n%d\n","#comment",WIDTH,HEIGHT,MAX_COLOR);

  // The number of batches may need to be adjusted along with the number of
  // devices to get the best results.
  unsigned int NUM_BATCHES=64;
  unsigned int BATCH_SIZE=HEIGHT/NUM_BATCHES;
  int queue, num_gpus, max_gpus = acc_get_num_devices(acc_device_nvidia);

  // NOTE: This region is to workaround timing irregularities due to the short
  // runtime of this example. In a real application, the overhead that this
  // region absorbs will be negligible.
#pragma omp parallel
  {
    num_gpus = omp_get_num_threads();
    int my_gpu = omp_get_thread_num();
    unsigned int num_batches_per_gpu = NUM_BATCHES/num_gpus;
    #pragma omp master
    {
      if ( num_gpus > max_gpus)
      {
        fprintf(stderr, "Trying to run on %d GPUs when only %d are available. Please reduce OMP_NUM_THREADS\n", num_gpus, max_gpus);
      }
    }

    // Sets the device for all interactions from this thread
    acc_set_device_num(my_gpu, acc_device_nvidia);
    acc_init(acc_device_nvidia);
#pragma acc parallel num_gangs(1)
    {
      my_gpu++; // Do nothing, just to try warming up the device.
    }
  }

  start=omp_get_wtime();
#pragma omp parallel
  {
    num_gpus = omp_get_num_threads();
    int my_gpu = omp_get_thread_num();
    unsigned int num_batches_per_gpu = NUM_BATCHES/num_gpus;
    #pragma omp master
    {
      if ( num_gpus > max_gpus)
      {
        fprintf(stderr, "Trying to run on %d GPUs when only %d are available. Please reduce OMP_NUM_THREADS\n", num_gpus, max_gpus);
      }
    }

    // Sets the device for all interactions from this thread
    acc_set_device_num(my_gpu, acc_device_nvidia);

    #pragma acc data create(image[0:HEIGHT*WIDTH])
    {
      queue = 1;
// Let OpenMP distribute the blocks to different threads. Because of the
// inherent load imbalance in this code, it will likely be necessary to tweak
// the schedule. A static, interleaved schedule is shown here to avoid uneven
// work distribution for more than 2 devices.
#pragma omp for schedule(static,1) firstprivate(queue)
    for(unsigned int batch=0;batch<NUM_BATCHES;batch++) {
      unsigned int ystart=batch*BATCH_SIZE;
      unsigned int yend=ystart+BATCH_SIZE;
      #pragma acc parallel loop collapse(2) present(image) async(batch%3+1)
      for(unsigned int y=ystart;y<yend;y++) {
        for(unsigned int x=0;x<WIDTH;x++) {
          image[y*WIDTH+x]=mandelbrot(x,y);
        }
      }
      #pragma acc update host(image[batch*BATCH_SIZE*WIDTH:WIDTH*BATCH_SIZE]) async(batch%3+1)
      //
      // Ensure we alternate queues on the device, regardless of the number of
      // devices or loop schedule. Avoiding the "0" queue, which may be special
      // on some devices.
      queue = queue%2 + 1;
    }

    // Wait for all work to complete
    #pragma acc wait

    }
  }
  end=omp_get_wtime();
  printf("Mandlebrot time: %f seconds\n", end-start);
  
  fwrite(image,sizeof(unsigned char),WIDTH*HEIGHT,fp);
  fclose(fp);
  free(image);
  return 0;
}
