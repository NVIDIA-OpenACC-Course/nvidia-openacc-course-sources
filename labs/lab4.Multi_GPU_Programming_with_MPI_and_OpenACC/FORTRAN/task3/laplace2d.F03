!  Copyright 2015 NVIDIA Corporation
!
!  Licensed under the Apache License, Version 2.0 (the "License");
!  you may not use this file except in compliance with the License.
!  You may obtain a copy of the License at
!
!      http://www.apache.org/licenses/LICENSE-2.0
!
!  Unless required by applicable law or agreed to in writing, software
!  distributed under the License is distributed on an "AS IS" BASIS,
!  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
!  See the License for the specific language governing permissions and
!  limitations under the License.

PROGRAM laplace2d
    USE openacc
    USE mpi
    IMPLICIT NONE
    INTEGER, PARAMETER :: n = 4096
    INTEGER, PARAMETER :: m = 4096
    INTEGER, PARAMETER :: iter_max = 1000
    REAL, PARAMETER :: tol = 1.0E-5
    REAL, PARAMETER :: pi = 2.0*ASIN(1.0)
    INTEGER :: i, j, jstart, jend, iter, mpi_rank, mpi_size, ngpus, devicenum, ierror, chunk_size, top, bottom
    REAL :: y0, error, runtime_serial, runtime, start, finish, globalerror
    LOGICAL, EXTERNAL :: check_results
    REAL, DIMENSION(n,m) :: a, a_ref, a_new
    
    mpi_rank = 0
    mpi_size = 1
    
    !Initialize MPI and determine rank and size
    CALL MPI_Init(ierror)
    CALL MPI_Comm_rank(MPI_COMM_WORLD,mpi_rank,ierror);
    CALL MPI_Comm_size(MPI_COMM_WORLD,mpi_size,ierror);
    
    a = 0.0
    a_ref = 0.0
    
    DO j = 1, m
        y0         = SIN( 2.0 * pi * j / (m-1) )
        a(1,j)     = y0
        a(n,j)     = y0
        a_ref(1,j) = y0
        a_ref(n,j) = y0
    END DO
    
#if _OPENACC
    ngpus=acc_get_num_devices( acc_device_nvidia )
    !choose device to use by this rank
    devicenum = MOD( mpi_rank, ngpus )
    call acc_set_device_num( devicenum, acc_device_nvidia )
    !Call acc_init after acc_set_device_num to avoid multiple contexts on device 0 in multi GPU systems
    call acc_init( acc_device_nvidia )
#endif
    !set first and last row to be processed by this rank.
    !Ensure correctness if m%size != 0
    chunk_size = CEILING( (1.0*m)/mpi_size )
    jstart = mpi_rank * chunk_size
    jend = jstart + chunk_size - 1
    
    !Do not process boundaries
    jstart = MAX( jstart, 2 )
    jend = MIN( jend, M-1 )

    IF ( mpi_rank == 0 ) THEN
        WRITE(*,"('Jacobi relaxation Calculation: ',I4,' x ',I4,' mesh')") n,m
        WRITE(*,*) 'Calculate reference solution and time serial execution.'
    END IF
    CALL cpu_time(start)
    CALL laplace2d_serial( n, m, iter_max, mpi_rank, tol, a_ref, a_new )
    CALL cpu_time(finish)
    runtime_serial = finish-start
    
    !Wait for all processes to ensure correct timing of the parallel version
    CALL MPI_Barrier( MPI_COMM_WORLD, ierror )
    
    IF ( mpi_rank == 0 ) THEN
        WRITE(*,*) 'Parallel execution.'
    END IF 
    
    CALL cpu_time(start)
    iter = 1
    error = 1.0
    !$acc data copy(a) create(a_new)
    DO WHILE ( error > tol .AND. iter <= iter_max )
        error = 0.0
        !$acc kernels
        DO j = jstart, jend
            DO i = 2, n-1
                a_new(i,j) = 0.25 * ( a(i+1,j) + a(i-1,j) + a(i,j-1) + a(i,j+1) )
                error = MAX( error, ABS( a_new(i,j) - a(i,j) ) )
            END DO
        END DO
        !$acc end kernels
        !Calculate global error across all ranks
        globalerror = 0.0;
        call MPI_Allreduce( error, globalerror, 1, MPI_REAL, MPI_MAX, MPI_COMM_WORLD, ierror );
        error = globalerror;
        
        !TODO: Split into halo and bulk part
        !$acc kernels
        DO j = jstart, jend
            DO i = 2, n-1
                a(i,j) = a_new(i,j)
            END DO
        END DO
        !$acc end kernels
        !TODO: Start bulk part asynchronously
        
        !Handle periodic boundary conditions and halo exchange with MPI
        top = mpi_rank-1
        IF ( mpi_rank == 0 ) THEN
            top = mpi_size-1
        END IF
        bottom = mpi_rank+1
        IF ( mpi_rank == mpi_size-1 ) THEN
            bottom = 0
        END IF
        
        !$acc host_data use_device( A )
            !1. Sent row jstart (first modified row) to top receive lower boundary (jend+1) from bottom
            CALL MPI_Sendrecv( a(1,jstart), n, MPI_REAL, top   , 0, a(1,jend+1), n, MPI_REAL, bottom, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror )

            !2. Sent row jend (last modified row) to bottom receive upper boundary (jstart-1) from top
            CALL MPI_Sendrecv( a(1,jend), n, MPI_REAL, bottom, 0, a(1,(jstart-1)), n, MPI_REAL, top   , 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror )
        !$acc end host_data
        
        !TODO: wait for bulk part
        
        IF ( mpi_rank == 0 .AND. ( iter == 1 .OR. MOD( iter, 100 ) == 0 ) ) THEN
            WRITE(*,"('  ',I4,' ',F8.6)") iter, error
        END IF
        
        iter = iter+1
    END DO
    !$acc end data
    !Wait for all processes to ensure correct timing of the parallel version
    CALL MPI_Barrier( MPI_COMM_WORLD, ierror )
    CALL cpu_time(finish)
    runtime = finish-start
    
    IF ( check_results( mpi_rank, jstart, jend, n, m, tol, a, a_ref ) ) THEN
        IF ( mpi_rank == 0 ) THEN
            WRITE(*,*) 'Num GPUs: ', mpi_size
            WRITE(*,"(I4,'x',I4,': 1 GPU: ',F8.4,' s ',I1,' GPUs: ',F8.4,' s, speedup: ',F8.2,' efficiency: ',F8.2)"),n,m,runtime_serial,mpi_size,runtime,runtime_serial/runtime,runtime_serial/(mpi_size*runtime)*100
        END IF
    END IF
    !Finalize MPI
    CALL MPI_Finalize(ierror)
END PROGRAM laplace2d
