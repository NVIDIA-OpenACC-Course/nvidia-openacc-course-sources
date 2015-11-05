/*
 *  Copyright 2015 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <math.h>
#include <string.h>
#include <stdio.h>
#ifdef _OPENACC
#include <openacc.h>
#endif /*_OPENACC*/
#include "common.h"

#include <mpi.h>

#define N 4096
#define M 4096

float A[N][M];
float Aref[N][M];
float Anew[N][M];

int main(int argc, char** argv)
{
    int iter_max = 1000;
    
    const float pi  = 2.0 * asinf(1.0f);
    const float tol = 1.0e-5f;

    int rank = 0;
    int size = 1;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    memset(A, 0, N * M * sizeof(float));
    memset(Aref, 0, N * M * sizeof(float));
    
    // set boundary conditions
    for (int j = 0; j < N; j++)
    {
        float y0     = sinf( 2.0 * pi * j / (N-1));
        A[j][0]      = y0;
        A[j][M-1]    = y0;
        Aref[j][0]   = y0;
        Aref[j][M-1] = y0;
    }
    
#if _OPENACC
    int ngpus=acc_get_num_devices(acc_device_nvidia);
    int devicenum=rank%ngpus;
    acc_set_device_num(devicenum,acc_device_nvidia);

    // Call acc_init after acc_set_device_num to avoid multiple contexts on device 0 in multi GPU systems
    acc_init(acc_device_nvidia);
#endif /*_OPENACC*/

    //TODO: set first and last row to be processed by this rank.
    int jstart = 1;
    int jend   = N-1;

    if ( rank == 0) printf("Jacobi relaxation Calculation: %d x %d mesh\n", N, M);

    if ( rank == 0) printf("Calculate reference solution and time serial execution.\n");
    StartTimer();
    laplace2d_serial( rank, iter_max, tol );
    double runtime_serial = GetTimer();

    //Wait for all processes to ensure correct timing of the parallel version
    MPI_Barrier( MPI_COMM_WORLD );
    if ( rank == 0) printf("Parallel execution.\n");
    StartTimer();
    int iter  = 0;
    float error = 1.0f;
    
    #pragma acc data copy(A) create(Anew)
    while ( error > tol && iter < iter_max )
    {
        error = 0.f;

        #pragma acc kernels
        for (int j = jstart; j < jend; j++)
        {
            for( int i = 1; i < M-1; i++ )
            {
                Anew[j][i] = 0.25f * ( A[j][i+1] + A[j][i-1]
                                     + A[j-1][i] + A[j+1][i]);
                error = fmaxf( error, fabsf(Anew[j][i]-A[j][i]));
            }
        }
        //TODO: Calculate global error across all ranks
        //MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_COMM_WORLD);
        
        #pragma acc kernels
        for (int j = jstart; j < jend; j++)
        {
            for( int i = 1; i < M-1; i++ )
            {
                A[j][i] = Anew[j][i];
            }
        }

        //Periodic boundary conditions
        //TODO: Handle periodic boundary conditions and halo exchange with MPI
        #pragma acc kernels
        for( int i = 1; i < M-1; i++ )
        {
                A[0][i]     = A[(N-2)][i];
                A[(N-1)][i] = A[1][i];
        }
        int top    = (rank == 0) ? (size-1) : rank-1;
        int bottom = (rank == (size-1)) ? 0 : rank+1;

        #pragma acc host_data use_device( A )
        {
            //TODO: 1. Sent row jstart (first modified row) to top receive lower boundary (jend) from bottom
            //MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, 0,
            //                       void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, 0,
            //                 MPI_COMM_WORLD, MPI_STATUS_IGNORE );

            //TODO: 2. Sent row (jend-1) (last modified row) to bottom receive upper boundary (jstart-1) from top
            //MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, 0,
            //                       void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, 0,
            //                 MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        }
        
        if(rank == 0 && (iter % 100) == 0) printf("%5d, %0.6f\n", iter, error);
        
        iter++;
    }
    MPI_Barrier( MPI_COMM_WORLD );
    double runtime = GetTimer();

    if (check_results( rank, jstart, jend, tol ) && rank == 0)
    {
        printf( "Num GPUs: %d\n", size );
        printf( "%dx%d: 1 GPU: %8.4f s, %d GPUs: %8.4f s, speedup: %8.2f, efficiency: %8.2f%\n", N,M, runtime_serial/ 1000.f, size, runtime/ 1000.f, runtime_serial/runtime, runtime_serial/(size*runtime)*100 );
    }
    MPI_Finalize();
    return 0;
}

#include "laplace2d_serial.h"
