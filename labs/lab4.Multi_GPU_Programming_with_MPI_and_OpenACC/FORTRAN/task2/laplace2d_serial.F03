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

SUBROUTINE laplace2d_serial(n, m, iter_max, mpi_rank,tol,a_ref, a_new)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: n, m, iter_max, mpi_rank
    REAL, INTENT(IN) :: tol
    REAL, DIMENSION(n,m), INTENT(INOUT) :: a_ref, a_new
    INTEGER :: i,j, iter
    REAL :: error
    
    iter = 1
    error = 1.0
    !$acc data copy(a_ref) create(a_new)
    DO WHILE ( error > tol .AND. iter <= iter_max )
        error = 0.0
        !$acc kernels
        DO j = 2, m-1
            DO i = 2, n-1
                a_new(i,j) = 0.25 * ( a_ref(i+1,j) + a_ref(i-1,j) + a_ref(i,j-1) + a_ref(i,j+1) )
                error = MAX( error, ABS( a_new(i,j) - a_ref(i,j) ) )
            END DO
        END DO
        
        DO j = 2, m-1
            DO i = 2, n-1
                a_ref(i,j) = a_new(i,j)
            END DO
        END DO
        
        DO i = 2, n-1
            a_ref(i,1) = a_ref(i,m-1)
            a_ref(i,m) = a_ref(i,2)
        END DO
        !$acc end kernels
        
        IF ( mpi_rank == 0 .AND. ( iter == 1 .OR. MOD( iter, 100 ) == 0 ) ) THEN
            WRITE(*,"('  ',I4,' ',F8.6)") iter, error
        END IF
        
        iter = iter+1
    END DO
    !$acc end data
END SUBROUTINE laplace2d_serial

LOGICAL FUNCTION check_results( mpi_rank, jstart, jend, n, m, tol, a, a_ref )
    USE mpi
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: mpi_rank, jstart, jend, n, m
    REAL, INTENT(IN) :: tol
    REAL, DIMENSION(n,m), INTENT(IN) :: a, a_ref
    INTEGER :: i,j,ierror
    LOGICAL :: no_errors, global_no_errors, mpi_is_initialized
    no_errors = .TRUE.
    j = jstart
    i = 1
    DO WHILE ( j <= jend .AND. no_errors )
        DO WHILE ( i <= n .AND. no_errors )
            IF ( ABS( a_ref(i,j) - a(i,j)) >= tol ) THEN
                WRITE(*,"('[MPI',I1,'] ERROR: a(',I4,',',I4,') = ',F8.6,' does not match ',F8.6,' (reference)')") mpi_rank,i,j,a(i,j),a_ref(i,j)
                no_errors = .FALSE.
            END IF
            i = i + 1
        END DO
        j = j + 1
    END DO
    
    CALL MPI_Initialized(mpi_is_initialized, ierror)
    IF ( mpi_is_initialized ) THEN
        global_no_errors = .FALSE.
        CALL MPI_ALLREDUCE(no_errors, global_no_errors, 1, MPI_LOGICAL, MPI_LAND, MPI_COMM_WORLD, ierror)
        no_errors = global_no_errors
    END IF
    check_results = no_errors
END FUNCTION check_results
