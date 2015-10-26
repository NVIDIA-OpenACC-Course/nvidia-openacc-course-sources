NVIDIA OpenACC Course - Lab 3
=============================

In this lab you will build upon the work from lab 2 to add explicit data
management directives, eliminating the need to use CUDA Unified Memory, and
optimize the `matvec` kernel using the OpenACC `loop` directive. If you have
not already completed lab 2, please go back and complete it before starting
this lab.

Step 1 - Express Data Movement
------------------------------
In the previous lab we used CUDA Unified Memory, which we enabled with the
`ta=tesla:managed` compiler option, to eliminate the need for data management
directives. Replace this compiler flag in the Makefile with `-ta=tesla` and try
to rebuild the code.

### C/C++
With the managed memory option removed the C/C++ version will fail to build
because the compiler will not be able to determine the sizes of some of the
arrays used in compute regions. You will see an error like the one below.

    PGCC-S-0155-Compiler failed to translate accelerator region (see -Minfo messages): Could not find allocated-variable index for symbol (main.cpp: 15)

### Fortran
The Fortran version of the code will build successfully and run, however the
tolerance value will be incorrect with the managed memory option removed.

    $ ./cg
     Rows:      8120601 nnz:    218535025
     Iteration:  0 Tolerance: 4.006700E+08
     Iteration: 10 Tolerance: 4.006700E+08
     Iteration: 20 Tolerance: 4.006700E+08
     Iteration: 30 Tolerance: 4.006700E+08
     Iteration: 40 Tolerance: 4.006700E+08
     Iteration: 50 Tolerance: 4.006700E+08
     Iteration: 60 Tolerance: 4.006700E+08
     Iteration: 70 Tolerance: 4.006700E+08
     Iteration: 80 Tolerance: 4.006700E+08
     Iteration: 90 Tolerance: 4.006700E+08
      Total Iterations:          100

We can correct these problems by explicitly declaring the data movement for the
arrays that we need on the GPU. In the associated lecture we discussed the
OpenACC structured `data` directive and the unstructured `enter data` and `exit
data` directives. Either approaced can be used to express the data locality in
this code, but the unstructured directives are probably cleaner to use.

### C/C++
In the `allocate_3d_poisson_matrix` function in matrix.h, add the following two
directives to the end of the function.


    #pragma acc enter data copyin(A)
    #pragma acc enter data copyin(A.row_offsets[:num_rows+1],A.cols[:nnz],A.coefs[:nnz])

The first directive copies the A structure to the GPU, which includes the
`num_rows` member and the pointers for the three member arrays. The second
directive then copies the three arrays to the device. Now that we've created
space on the GPU for these arrays, it's necessary to clean up the space when
we're done. In the `free_matrix` function, att the following directives
immediately before the calls to `free`.

    #pragma acc exit data delete(A.row_offsets,A.cols,A.coefs)
    #pragma acc exit data delete(A)
      free(row_offsets);
      free(cols);
      free(coefs);

Notice that we are performing the operations in the reverse order. First we are
deleting the 3 member arrays from the device, then we are deleting the
structure containing those arrays. It's also critical that we place our pragmas
*before* the arrays are freed on the host, otherwise the `exit data` directives
will fail.

Now go into `vector.h` and do the same thing in `allocate_vector` and
`free_vector` with the structure `v` and its member array `v.coefs`. If you try
to build again at this point, the code will still fail to build because we
haven't told our compute regions that the data is already present on the
device, so the compiler is still trying to determine the array sizes itself.
Now go to the compute regions (`kernels` or `parallel loop`) in
`matrix_functions.h` and `vector_functions.h` and use the `present` clause to
inform the compiler that the arrays are already on the device. Below is an
example for `matvec`.

    #pragma acc kernels present(row_offsets,cols,Acoefs,xcoefs,ycoefs)

The application should now build and run on the GPU, but is no longer getting
correct results. This is because we've put the arrays on the device, but we've
failed to copy the input data into these arrays. Add the following directive to
the end of `initialize_vector` function.

    #pragma acc update device(v.coefs[:v.n])

This will copy the data entered in the host array by this function to the GPU
copy of the array. 

### Fortran
To make the application return correct answers again, it will be necessary to
add explicit data management directives. This could be done using either the
structured `data` directives or unstructured `enter data` and `exit data`
directives, as discussed in the lecture. Since this program has clear routines
for allocating and initializing the data structures and also deallocating,
we'll use the unstructured directives.

The `allocate_3d_poission_matrix` in matrix.F90 handles allocating and
initializing the primary array. At the end of this routine, add the following
directive for copying the three arrays in the matrix type to the device.

    !$acc enter data copyin(arow_offsets,acols,acoefs)

These three arrays can be copied in seperate `enter data` directives as well.
Notice that because Fortran arrays are self-describing, it's unnecessary to
provide the array bounds, although it would be safe to do so as well. Since
we've allocated these arrays on the device, they should be removed from the
device when we are done with them as well. In the `free_matrix` subroutine of
matrix.F90 add the following directive.

    !$acc exit data delete(arow_offsets,acols,acoefs)
    deallocate(arow_offsets)
    deallocate(acols)
    deallocate(acoefs)

Notice that the `exit data` directive appears before the `deallocate`
statement. Because the OpenACC programming model assumes we always begin and
end execution on the host, it's necessary to remove arrays from the device
before freeing them on the host to avoid an error or crash. Now go add i
`enter data` and `exit data` directives to vector.F90 as well. Notice that the
`allocate_vector` routine only allocates the array, but does not initialize it,
so `copyin` may be replaced with `create` on the `enter data` directive.

If we build and run the application at this point we should see our tolerance
changing once again, but the answers will still be incorrect. Next let go to
each compute directive (`kernels` or `parallel loop`) in matrix.F90 and
vector.F90 and inform the compiler that the arrays used in those regions are
already present on the device. Below is an example from matrix.F90.

    !$acc kernels present(arow_offsets,acols,acoefs,x,y)

At this point the compiler knows that it does not need to be concerned with
data movement in our compute regions, but we're still getting the wrong answer.
The last change we need to make is to make sure that we're copying the input
data to the device before execution. In vector.F90 add the following directive
to the end of `initialize_vector`.

    vector(:) = value
    !$acc update device(vector)

Now that we have the correct input data on the device the code should run
correctly once again.

---

(NOTE for C/C++ and Fortran: One could also parallelize the loop in
`initialize_vector` on the GPU, but we choose to use the `update` directive
here to illustrate how this directive is used.)

Step 2 - Profile The Application
--------------------------------
Just as in the last lab, we'll use the NVIDIA Visual Profiler to profile our
application. 

PASTE INSTRUCTIONS TO OPEN NVVP

Once you've opened Visual Profiler, go to File and then New Session. 

In the lower left, press the "Examine GPU Usage" button.

Step 3 - Optimize Loops
-----------------------
Given that the analysis above shows poor resource utilization, let's apply our
knowledge of the code to help the compiler make better decisions about how to
parallelize our loops. We know from the `allocate_3d_poission_matrix` routine
that the most non-zero elements we'll have per row is 27. This means that with
the compiler-selected vector length of 128, 101 vector lanes (threads) will go
unused. Let's tell the compiler to choose a better vector length for these
loops. 

On an NVIDIA GPU the vector length must be a multiple of the *warp size* of the
GPU, which on all NVIDIA GPUs to-date is 32. This means that the closest vector
length we can choose is 32. Depending on whether the code uses `kernels` or 
`parallel loop`, we can specify the vector length one of two ways.

### Kernels
When using the `kernels` directive, the vector length is given by adding
`kernels(32)` to the loop we want to use as the `vector` loop. So for our
`matvec` loops, we'd apply the vector length as shown below.

#### C/C++
    #pragma acc kernels present(row_offsets,cols,Acoefs,xcoefs,ycoefs)
      {
    #pragma acc loop 
        for(int i=0;i<num_rows;i++) {
          double sum=0;
          int row_start=row_offsets[i];
          int row_end=row_offsets[i+1];
          #pragma acc loop device_type(nvidia) vector(32)
          for(int j=row_start;j<row_end;j++) {
            unsigned int Acol=cols[j];
            double Acoef=Acoefs[j];
            double xcoef=xcoefs[Acol];
            sum+=Acoef*xcoef;
          }
          ycoefs[i]=sum;
        }
      }

#### Fortran
    !$acc kernels present(arow_offsets,acols,acoefs,x,y)
    do i=1,a%num_rows
      tmpsum = 0.0d0
      row_start = arow_offsets(i)
      row_end   = arow_offsets(i+1)-1
      !$acc loop device_type(nvidia) vector(32)
      do j=row_start,row_end
        acol = acols(j)
        acoef = acoefs(j)
        xcoef = x(acol)
        tmpsum = tmpsum + acoef*xcoef
      enddo
      y(i) = tmpsum
    enddo
    !$acc end kernels

### Parallel Loop
When using `parallel loop` the vector length is given at the top of the loop,
as shown below.

#### C/C++
    #pragma acc parallel loop present(row_offsets,cols,Acoefs,xcoefs,ycoefs) \
            device_type(nvidia) vector_length(32)
      for(int i=0;i<num_rows;i++) {
        double sum=0;
        int row_start=row_offsets[i];
        int row_end=row_offsets[i+1];
    #pragma acc loop reduction(+:sum) device_type(nvidia) vector
        for(int j=row_start;j<row_end;j++) {
          unsigned int Acol=cols[j];
          double Acoef=Acoefs[j];
          double xcoef=xcoefs[Acol];
          sum+=Acoef*xcoef;
        }
        ycoefs[i]=sum;
      }

#### Fortran
    !$acc parallel loop private(tmpsum,row_start,row_end) &
    !$acc& present(arow_offsets,acols,acoefs,x,y)         &
    !$acc& device_type(nvidia) vector_length(32)
    do i=1,a%num_rows
      tmpsum = 0.0d0
      row_start = arow_offsets(i)
      row_end   = arow_offsets(i+1)-1
      !$acc loop reduction(+:tmpsum) device_type(nvidia) vector
      do j=row_start,row_end
        acol = acols(j)
        acoef = acoefs(j)
        xcoef = x(acol)
        tmpsum = tmpsum + acoef*xcoef
      enddo
      y(i) = tmpsum
    enddo

---

Notice that the above code adds the `device_type(nvidia)` clause to the
affected loops. Because we only want this optimization to be applied to NVIDIA
GPUs, we've protected that optimization with a `device_type` clause and allowed
the compiler to determine the best value on other platforms.

Conclusion
----------
