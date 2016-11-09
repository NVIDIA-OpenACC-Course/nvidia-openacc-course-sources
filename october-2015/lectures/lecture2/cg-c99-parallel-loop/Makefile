CC=pgc++
CFLAGS=-fast -acc -ta=tesla:managed -Minfo=accel

all: 
	${CC} ${CFLAGS} main.cpp -o cg
clean:
	rm -Rf cg pgprof* gprof.out *.o core
