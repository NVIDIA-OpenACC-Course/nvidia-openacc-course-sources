all: run

laplace2d: laplace2d.F03 laplace2d_serial.F03 Makefile
	mpifort -fast -acc -ta=tesla -Minfo=accel laplace2d.F03 laplace2d_serial.F03 -o laplace2d

clean:
	rm -f laplace2d laplace2d.solution *.o

run: laplace2d
	#TODO: Start with MPI run
	./laplace2d

profile: laplace2d
	mpirun -np 4 nvprof -o laplace2d.%q{OMPI_COMM_WORLD_RANK}.nvvp ./laplace2d