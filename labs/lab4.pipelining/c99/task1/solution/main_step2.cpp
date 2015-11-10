#include <cstdio>
#include <cstdlib>
#include <fstream>
#include "constants.h"

using namespace std;

#pragma acc routine seq
unsigned char mandelbrot(int Px, int Py);

int main() {
  
  size_t bytes=WIDTH*HEIGHT*sizeof(unsigned int);
  unsigned char *image=(unsigned char*)malloc(bytes);
  int num_blocks = 8, block_size = (HEIGHT/num_blocks)*WIDTH;
  FILE *fp=fopen("image.pgm","wb");
  fprintf(fp,"P5\n%s\n%d %d\n%d\n","#comment",WIDTH,HEIGHT,MAX_COLOR);

#pragma acc data create(image[WIDTH*HEIGHT])
  for(int block = 0; block < num_blocks; block++ ) {
    int start = block * (HEIGHT/num_blocks),
        end   = start + (HEIGHT/num_blocks);
#pragma acc update device(image[block*block_size:block_size]) async(block)
#pragma acc parallel loop async(block)
    for(int y=start;y<end;y++) {
      for(int x=0;x<WIDTH;x++) {
        image[y*WIDTH+x]=mandelbrot(x,y);
      }
    }
#pragma acc update self(image[block*block_size:block_size]) async(block)
  }
#pragma acc wait
  
  fwrite(image,sizeof(unsigned char),WIDTH*HEIGHT,fp);
  fclose(fp);
  free(image);
  return 0;
}
