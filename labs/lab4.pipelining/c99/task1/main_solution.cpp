#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <omp.h>
#include "constants.h"

using namespace std;

#pragma acc routine seq
unsigned char mandelbrot(int Px, int Py);

int main() {
  
  size_t bytes=WIDTH*HEIGHT*sizeof(unsigned int);
  unsigned char *image=(unsigned char*)malloc(bytes);
  int num_blocks = 8;
  FILE *fp=fopen("image.pgm","wb");
  fprintf(fp,"P5\n%s\n%d %d\n%d\n","#comment",WIDTH,HEIGHT,MAX_COLOR);
  double st = omp_get_wtime();

  for(int block = 0; block < num_blocks; block++ ) {
    int start = block * (HEIGHT/num_blocks),
        end   = start + (HEIGHT/num_blocks);
#pragma acc parallel loop
    for(int y=start;y<end;y++) {
      for(int x=0;x<WIDTH;x++) {
        image[y*WIDTH+x]=mandelbrot(x,y);
      }
    }
  }
  
  double et = omp_get_wtime();
  printf("Time: %lf seconds.\n", (et-st));
  fwrite(image,sizeof(unsigned char),WIDTH*HEIGHT,fp);
  fclose(fp);
  free(image);
  return 0;
}
