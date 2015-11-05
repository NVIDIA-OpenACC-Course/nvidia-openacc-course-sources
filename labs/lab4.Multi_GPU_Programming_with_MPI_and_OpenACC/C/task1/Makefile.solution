all: run

laplace2d: laplace2d.c common.h laplace2d_serial.h Makefile
	mpicc -fast -acc -ta=nvidia laplace2d.c -o laplace2d

clean:
	rm -f laplace2d laplace2d.solution laplace2d.*.nvvp

run: laplace2d
	mpirun -np 4 ./laplace2d

profile: laplace2d
	mpirun -np 4 nvprof -o laplace2d.%q{OMPI_COMM_WORLD_RANK}.nvvp ./laplace2d

laplace2d.solution: laplace2d.solution.c common.h laplace2d_serial.h Makefile
	mpicc -fast -acc -ta=nvidia laplace2d.solution.c -o laplace2d.solution

solution: laplace2d.solution
	mpirun -np 4 ./laplace2d.solution

