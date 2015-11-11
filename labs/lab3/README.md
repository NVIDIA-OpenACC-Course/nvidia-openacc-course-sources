NVIDIA OpenACC Course - Lab 3
=============================

In this lab you will build upon the work from lab 2 to add explicit data
management directives, eliminating the need to use CUDA Unified Memory, and
optimize the `matvec` kernel using the OpenACC `loop` directive. If you have
not already completed lab 2, please go back and complete it before starting
this lab.

Step 0 - Building the code
--------------------------

Makefiles have been provided for building both the C and Fortran versions of
the code. Change directory to your language of choice and run the `make`
command to build the code.

### C/C++

```
$ cd ~/c99
$ make
```
    
### Fortran

```
$ cd ~/f90
$ make
```
    
This will build an executable named `cg` that you can run with the `./cg`
command. You may change the options passed to the compiler by modifying the
`CFLAGS` variable in `c99/Makefile` or `FCFLAGS` in `f90/Makefile`. You should
not need to modify anything in the Makefile except these compiler flags.

Step 1 - Step 1 - Express Data Movement
---------------------------------------

In the previous lab we used CUDA Unified Memory, which we enabled with the
`ta=tesla:managed` compiler option, to eliminate the need for data management
directives. Replace this compiler flag in the Makefile with `-ta=tesla` and try
to rebuild the code.

### C/C++
With the managed memory option removed the C/C++ version will fail to build
because the compiler will not be able to determine the sizes of some of the
arrays used in compute regions. You will see an error like the one below.

    PGCC-S-0155-Compiler failed to translate accelerator region (see -Minfo
messages): Could not find allocated-variable index for symbol (main.cpp: 15)

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

---

We can correct both of these problems by explicitly declaring the data movement
for the arrays that we need on the GPU. In the associated lecture we discussed
the OpenACC structured `data` directive and the unstructured `enter data` and
`exit data` directives. Either approaced can be used to express the data
locality in this code, but the unstructured directives are probably cleaner to
use.

### C/C++
In the `allocate_3d_poisson_matrix` function in matrix.h, add the following two
directives to the end of the function.


    #pragma acc enter data copyin(A)
    #pragma acc enter data copyin(A.row_offsets[:num_rows+1],A.cols[:nnz],A.coefs[:nnz])

The first directive copies the A structure to the GPU, which includes the
`num_rows` member and the pointers for the three member arrays. The second
directive then copies the three arrays to the device. Now that we've created
space on the GPU for these arrays, it's necessary to clean up the space when
we're done. In the `free_matrix` function, add the following directives
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
`free_vector` with the structure `v` and its member array `v.coefs`. Because 
we are copying the arrays to the device before they have been populated with 
data, use the `create` data clause, rather than `copyin`.

If you try
to build again at this point, the code will still fail to build because we
haven't told our compute regions that the data is already present on the
device, so the compiler is still trying to determine the array sizes itself.
Now go to the compute regions (`kernels` or `parallel loop`) in
`matrix_functions.h` and `vector_functions.h` and use the `present` clause to
inform the compiler that the arrays are already on the device. Below is an
example for `matvec`.

    #pragma acc kernels present(row_offsets,cols,Acoefs,xcoefs,ycoefs)

Once you have added the `present` clause to all three compute regions, the 
application should now build and run on the GPU, but is no longer getting
correct results. This is because we've put the arrays on the device, but we've
failed to copy the input data into these arrays. Add the following directive to
the end of `initialize_vector` function.

    #pragma acc update device(v.coefs[:v.n])

This will copy the data now in the host array to the GPU
copy of the array. With this data now correctly copied to the GPU, the code
should run to completion and give the same results as before.

### Fortran
To make the application return correct answers again, it will be necessary to
add explicit data management directives. This could be done using either the
structured `data` directives or unstructured `enter data` and `exit data`
directives, as discussed in the lecture. Since this program has clear routines
for allocating and initializing the data structures and also deallocating,
we'll use the unstructured directives to make the code easy to understand.

The `allocate_3d_poisson_matrix` in matrix.F90 handles allocating and
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
before freeing them on the host to avoid an error or crash. Now go add 
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

*(NOTE for C/C++ and Fortran: One could also parallelize the loop in
`initialize_vector` on the GPU, but we choose to use the `update` directive
here to illustrate how this directive is used.)*

Step 2 - Optimize Loops - Vector Length
---------------------------------------

Now that we're running on the GPU and getting correct answers , let's apply our
knowledge of the code to help the compiler make better decisions about how to
parallelize our loops. We know from the `allocate_3d_poisson_matrix` routine
that the most non-zero elements we'll have per row is 27. By examining the
compiler output, as shown below, we know that the compiler chose a vector
length of 128 for the `matvec` loops. This means that with the
compiler-selected vector length of 128, 101 vector lanes (threads) will go
unused. Let's tell the compiler to choose a better vector length for these
loops.

    matvec(const matrix &, const vector &, const vector &):
          8, include "matrix_functions.h"
              15, Generating present(row_offsets[:],cols[:],Acoefs[:],xcoefs[:],ycoefs[:])
              16, Loop is parallelizable
                  Accelerator kernel generated
                  Generating Tesla code
                  16, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
              20, Loop is parallelizable

On an NVIDIA GPU the vector length must be a multiple of the *warp size* of the
GPU, which on all NVIDIA GPUs to-date is 32. This means that the closest vector
length we can choose is 32. Depending on whether the code uses `kernels` or
`parallel loop`, we can specify the vector length one of two ways.

### Kernels
When using the `kernels` directive, the vector length is given by adding
`vector(32)` to the loop we want to use as the `vector` loop. So for our
`matvec` loops, we'd apply the vector length as shown below.

#### C/C++
    #pragma acc kernels present(row_offsets,cols,Acoefs,xcoefs,ycoefs)
      {
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
When using `parallel loop` the vector length is given at the top of the region,
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
the compiler to determine the best value on other platforms. Now that we've
adjusted the vector length to fit the problem, let's profile the code again to
see how well it's performing. Using Visual Profiler, let's see if we can find a
way to further improve performance.

The folders `intermediate.kernels` and `intermediate.parallel` contain the
correct code for the end of this step. If you have any trouble, use the code in
one of these folders to help yourself along.

Step 3 - Optimize Loops - Profile The Application
-------------------------------------------------

Just as in the last lab, we'll use the NVIDIA Visual Profiler to profile our
application.

- If you are doing this lab on your own machine, either launch Visual Profiler
  from its application link or via the `nvvp` command.

Once Visual Profiler has started, create a new session by selecting *File ->
New Session*. Then select the executable that you built by pressing the
*Browse* button next to *File*, browse to your working directory, select the
`cg` executable, and then press *Next*. On the next screen press *Finish*.
Visual Profiler will run for several seconds to collect a GPU timeline and
begin its *guided analysis*.

In the lower left, press the "Examine GPU Usage" button. You may need to
enlarge the bottom panel of the screen by grabbing just below the horizontal
scroll bar at the middle of the window and dragging it up until the button is
visial. After this runs, click on "Examine Individual Kernels" and select the
top kernel in the table. After selecting the top kernel, press the "Perform
Kernel Analysis" button to gather further performance information about this
kernel and wait while Visual Profiler collects additional data ***(this make
take
several minutes)***. When this completes, press "Perform Latency Analysis". The 
screenshot below shows Visual Profiler at this step.

![NVIDIA Visual Profiler Limited By Block Size](lab3-nvvp-block-limit.png)

Visual Profiler is telling us that the performance of the matvec kernel is
limited by the amount of parallelism in each gang (referred to as *"block
size"* in CUDA).  Scrolling down in the *Results* section I see that the
*Occupancy* is 25%. Occupancy is a measure of how much parallelism is running
on the GPU versus how much theoretically could be running. 25% occupancy
indicates that resources are sitting idle due to the size of the blocks
(OpenACC gangs).

(***Note:*** *100% occupancy is not necessary for high performance, but
occupancy below 50% is frequently an indicator that optimization is possible)

Scrolling further down in the *Results* section we reach the *Block Limit*
metric, which will be highlighted in red. This is shown in the screenshot
below.

![NVIDIA Visual Profiler Occupancy Screenshot](lab3-nvvp-occupancy.png)
 
This table is showing us that the GPU *streaming multiprocessor (SM)* can
theoretically run 64 *warps* (groups of 32 threads), but only has 16 to run.
Looking at the *Warps/Block* and *Threads/Block* rows of the table, we see that
each block contains 1 warp, or 32 threads, although it could run many more.
This is because we've told the compiler to use a vector length of 32. As a
reminder, in OpenACC many *gangs* run independently of each other, each gang
has 1 or more *workers*, each of which operates on a *vector*. With a vector
length of 32, we'll need to add workers in order to increase the work per gang.
Now we need to inform the compiler to give each gang more work by using
*worker* parallelism.

Step 4 - Optimize Loops - Increase Parallelism
----------------------------------------------

To increase the parallelism in each OpenACC gang, we'll use the worker level of
parallelism to operate on multiple vectors within each gang. On an NVIDIA GPU
the *vector length X number of workers* must be a multiple of 32 and no larger
than 1024, so let's experiment with increasing the number of workers. From just
1 worker up to 32. We want the outermost loop to be divided among gangs and
workers, so we'll specify that it is an gang *and* worker loop. By only
specifying the number of workers, we allow the compiler to generate enough
gangs to use up the rest of the loop iterations applying worker parallelism. 


### Kernels
When using the `kernels` directive, use the `loop` directive to specify that
the outer loop should be a *gang* and *worker* loop with 32 workers as shown
below. Experiment with the number of workers to find the best value.

#### C/C++
    #pragma acc kernels present(row_offsets,cols,Acoefs,xcoefs,ycoefs)
      {
    #pragma acc loop device_type(nvidia) gang worker(32)
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
    !$acc loop device_type(nvidia) gang worker(32)
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
When using the `parallel loop` directive, use `gang` and `worker` to specify
that the outer loop should be a *gang* and *worker* loop and then add
`num_workers(32)` to specify 32 workers, as shown below. Experiment with 
the number of workers to find the best value.

#### C/C++
    #pragma acc parallel loop present(row_offsets,cols,Acoefs,xcoefs,ycoefs) \
            device_type(nvidia) gang worker vector_length(32) num_workers(32)
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
    !$acc& device_type(nvidia) gang worker num_workers(32) vector_length(32)
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

After experimenting with the number of workers, performance should be 
similar to the table below.

| Workers |  K40    |  Qwiklab  |
|---------|----------|-----------|
|    1    |          | 23.818328 |
|    2    | 61.03544 | 13.19415  |
|    4    | 31.36616 | 8.834735  |
|    8    | 16.71916 | 9.030089  |
|   16    | 8.81069  | 9.464214  |
|   32    | 6.488389 | 10.400797 |


Conclusion
----------

In this lab we started with a code that relied on CUDA Unified Memory to handle
data movement and added explicit OpenACC data locality directives. This makes
the code portable to any OpenACC compiler and accelerators that may not have
Unified Memory. We used both the unstructured data directives and the `update`
directive to achieve this.

Next we profiled the code to determine how it could run more efficiently on the
GPU we're using. We used our knowledge of both the application and the hardware
to find a loop mapping that ran well on the GPU, achieving a 2-4X speed-up over
our starting code.

The table below shows runtime for each step of this lab on an NVIDIA Tesla K40 
and on the Qwiklabs GPUs.

| Step             | K40       | Qwiklab GPU | 
| ---------------- | --------- | ----------- |
| Unified Memory   | 8.458172  | 32.084347   |
| Explicit Memory  | 8.459754  | 33.251878   | 
| Vector Length 32 | 11.656281 | 23.83046    |
| Final Code       | 4.802727  | 8.834735    | 
