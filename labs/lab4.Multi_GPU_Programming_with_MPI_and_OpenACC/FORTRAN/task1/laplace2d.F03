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
    !TODO: Use mpi
    IMPLICIT NONE
    INTEGER, PARAMETER :: n = 4096
    INTEGER, PARAMETER :: m = 4096
    INTEGER, PARAMETER :: iter_max = 1000
    REAL, PARAMETER :: tol = 1.0E-5
    REAL, PARAMETER :: pi = 2.0*ASIN(1.0)
    INTEGER :: i, j, jstart, jend, iter, mpi_rank, mpi_size, ngpus, devicenum, ierror
    REAL :: y0, error, runtime_serial, runtime, start, finish
    LOGICAL, EXTERNAL :: check_results
    REAL, DIMENSION(n,m) :: a, a_ref, a_new
    
    mpi_rank = 0
    mpi_size = 1
    
    !TODO: Initialize MPI and determine rank and size
    !call MPI_INIT(IERROR)
    !call MPI_COMM_RANK(MPI_COMM_WORLD, RANK, IERROR)
    !call MPI_COMM_SIZE(MPI_COMM_WORLD, SIZE, IERROR)
    
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
    !TODO: choose device to use by this rank
    devicenum = 0
    call acc_set_device_num( devicenum, acc_device_nvidia )
    !Call acc_init after acc_set_device_num to avoid multiple contexts on device 0 in multi GPU systems
    call acc_init( acc_device_nvidia )
#endif
    
    jstart = 2
    jend = m-1
    
    IF ( mpi_rank == 0 ) THEN
        WRITE(*,"('Jacobi relaxation Calculation: ',I4,' x ',I4,' mesh')") n,m
        WRITE(*,*) 'Calculate reference solution and time serial execution.'
    END IF
    CALL cpu_time(start)
    CALL laplace2d_serial( n, m, iter_max, mpi_rank, tol, a_ref, a_new )
    CALL cpu_time(finish)
    runtime_serial = finish-start
    
    !TODO: Wait for all processes to ensure correct timing of the parallel version
    !call MPI_BARRIER(MPI_COMM_WORLD, IERROR)
    
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
        
        !$acc kernels
        DO j = jstart, jend
            DO i = 2, n-1
                a(i,j) = a_new(i,j)
            END DO
        END DO
        !$acc end kernels
        
        !Periodic boundary conditions
        !$acc kernels
        DO i = 2, n-1
            a(i,1) = a(i,m-1)
            a(i,m) = a(i,2)
        END DO
        !$acc end kernels
        
        IF ( mpi_rank == 0 .AND. ( iter == 1 .OR. MOD( iter, 100 ) == 0 ) ) THEN
            WRITE(*,"('  ',I4,' ',F8.6)") iter, error
        END IF
        
        iter = iter+1
    END DO
    !$acc end data
    !TODO: Wait for all processes to ensure correct timing of the parallel version
    !call MPI_BARRIER(MPI_COMM_WORLD, IERROR)
    CALL cpu_time(finish)
    runtime = finish-start
    
    IF ( check_results( mpi_rank, jstart, jend, n, m, tol, a, a_ref ) ) THEN
        IF ( mpi_rank == 0 ) THEN
            WRITE(*,*) 'Num GPUs: ', mpi_size
            WRITE(*,"(I4,'x',I4,': 1 GPU: ',F8.4,' s ',I1,' GPUs: ',F8.4,' s, speedup: ',F8.2,' efficiency: ',F8.2)"),n,m,runtime_serial,mpi_size,runtime,runtime_serial/runtime,runtime_serial/(mpi_size*runtime)*100
        END IF
    END IF
    !TODO: Finalize MPI
    !call MPI_FINALIZE(IERROR)
END PROGRAM laplace2d
