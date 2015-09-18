NVIDIA OpenACC Course - Lab 2
=============================

In this lab you will profile the provided application using either NVIDIA
nvprof or gprof and the PGI compiler. After profiling the application, you will
use OpenACC to express the parallelism in the 3 most time-consumining routines.
You will use CUDA Unified Memory and the PGI "managed" option to manage host
and device memories for you. You may use either the `kernels` or `parallel loop` 
directives to express the parallelism in the code. Versions of the code
have been provided in C99 and Fortran 90.

**Hint** You should repeat steps 2 and 3 for each function identified in step 1
in order of function importance. Gather a new GPU profile each time and observe
how the profile changes after each step.

Step 1 - Identify Parallelism
-----------------------------

Step 2 - Express Parallelism
-----------------------------

Step 3 - Re-Profile Application
-------------------------------

Conclusion
----------
After completing the above steps for each of the 3 important routines your
application should show a speed-up over the unaccelerated version. You an
verify this by removing the `-ta` flag from your compiler options. 

Bonus
-----
1. Rebuild the application replacing `ta=tesla:managed` with `-ta=multicore`.
This will change the target accelerator from the GPU to the multicore CPU. Does
the code speed-up over the original by using CPU parallelism as well?
2. If you used the `kernels` directive to express the parallelism in the code,
try again with the `parallel loop` directive. Remember, you will need to take
responsibility of identifying any reductions in the code. If you used 
`parallel loop`, try using `kernels` instead and observe the differences both in
developer effort and performance.
