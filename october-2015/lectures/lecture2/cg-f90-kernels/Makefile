FC=pgfortran
FCFLAGS=-fast -ta=tesla:managed -Minfo=accel

cg: matrix.o vector.o main.F90
	${FC} -o $@ ${FCFLAGS} main.F90 vector.o matrix.o

matrix.o: matrix.F90
	${FC} ${FCFLAGS} matrix.F90 -c -o $@

vector.o: vector.F90
	${FC} ${FCFLAGS} vector.F90 -c -o $@

clean:
	rm -Rf cg pgprof* *.o *.mod core
