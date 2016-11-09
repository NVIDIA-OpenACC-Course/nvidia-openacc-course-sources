CC=pgc++
CFLAGS=-fast -acc -ta=tesla:pin -Minfo=all

all: threaded_mandelbrot unthreaded_mandelbrot
threaded_mandelbrot: mandelbrot.cpp main.cpp
	${CC} ${CFLAGS} -mp -o $@ -lrt $^
unthreaded_mandelbrot: mandelbrot.cpp main.cpp
	${CC} ${CFLAGS} -o $@ -lrt $^ 

clean:
	rm -Rf threaded_mandelbrot unthreaded_mandelbrot image.pgm *.o
