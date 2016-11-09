all: run

laplace2d: laplace2d.c common.h laplace2d_serial.h Makefile
	#TODO: Use MPI compiler wrapper
	pgcc -fast -acc -ta=nvidia laplace2d.c -o laplace2d

clean:
	rm -f laplace2d laplace2d.solution laplace2d.nvvp

run: laplace2d
	#TODO: Start with MPI run
	./laplace2d

profile: laplace2d
	nvprof -o laplace2d.nvvp ./laplace2d
