module vector_mod
  implicit none
  public :: initialize_vector,allocate_vector,free_vector
  public :: dot, waxpby
  contains
  subroutine initialize_vector(vector,value)
    implicit none
    real(8), intent(out) :: vector(:)
    real(8), intent(in)  :: value
    vector(:) = value
  end subroutine
  subroutine allocate_vector(vector,length)
    implicit none
    real(8), allocatable :: vector(:)
    integer              :: length
    allocate(vector(length))
  end subroutine allocate_vector
  subroutine free_vector(vector)
    implicit none
    real(8), allocatable :: vector(:)
    deallocate(vector)
  end subroutine
  function dot(x, y)
    implicit none
    real(8), intent(in) :: x(:), y(:)
    real(8)             :: dot, tmpsum
    integer             :: i, length

    length = size(x)
    tmpsum = 0.0
    !$acc parallel loop reduction(+:tmpsum)
    do i=1,length
      tmpsum = tmpsum + x(i)*y(i)
    enddo

    dot = tmpsum
  end function
  subroutine waxpby(alpha, x, beta, y, w)
    implicit none
    real(8), intent(in)  :: alpha, beta, x(:), y(:)
    real(8), intent(out) :: w(:)
    integer             :: i, length

    length = size(x)
    !$acc parallel loop
    do i=1,length
      w(i) = alpha*x(i) + beta*y(i)
    enddo

  end subroutine
end module vector_mod
