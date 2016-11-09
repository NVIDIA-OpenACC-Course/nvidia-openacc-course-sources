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

#ifndef LAPLACE2D_SERIAL_H
#define LAPLACE2D_SERIAL_H

void laplace2d_serial( int rank, int iter_max, float tol )
{
    int iter  = 0;
    float error = 1.0f;
    #pragma acc data copy(Aref) create(Anew)
    while ( error > tol && iter < iter_max )
    {
        error = 0.f;

#pragma acc kernels
        for( int j = 1; j < N-1; j++)
        {
            for( int i = 1; i < M-1; i++ )
            {
                Anew[j][i] = 0.25f * ( Aref[j][i+1] + Aref[j][i-1]
                                     + Aref[j-1][i] + Aref[j+1][i]);
                error = fmaxf( error, fabsf(Anew[j][i]-Aref[j][i]));
            }
        }

#pragma acc kernels
        for( int j = 1; j < N-1; j++)
        {
            for( int i = 1; i < M-1; i++ )
            {
                Aref[j][i] = Anew[j][i];
            }
        }

        //Periodic boundary conditions
#pragma acc kernels
        for( int i = 1; i < M-1; i++ )
        {
                Aref[0][i]     = Aref[(N-2)][i];
                Aref[(N-1)][i] = Aref[1][i];
        }

        if(rank == 0 && (iter % 100) == 0) printf("%5d, %0.6f\n", iter, error);

        iter++;
    }

}

int check_results( int rank, int jstart, int jend, float tol )
{
    int result_correct = 1;
    for( int j = jstart; j < jend && (result_correct == 1); j++)
    {
        for( int i = 1; i < M-1 && (result_correct == 1); i++ )
        {
            if ( fabs ( Aref[j][i] - A[j][i] ) >= tol )
            {
                printf("[MPI%d] ERROR: A[%d][%d] = %f does not match %f (reference)\n", rank, j,i, A[j][i], Aref[j][i]);
                result_correct = 0;
            }
        }
    }
#ifdef MPI_VERSION
    int global_result_correct = 0;
    MPI_Allreduce( &result_correct, &global_result_correct, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD );
    result_correct = global_result_correct;
#endif //MPI_VERSION
    return result_correct;
}

#endif // LAPLACE2D_SERIAL_H
