#pragma once
#include<cmath>

struct vector {
  unsigned int n;
  double *coefs;
};

void allocate_vector(vector &v, unsigned int n) {
  v.n=n;
  v.coefs=(double*)malloc(n*sizeof(double));
}

void free_vector(vector &v) {
  double *vcoefs=v.coefs;
  free(v.coefs);

}

void initialize_vector(vector &v,double val) {

  for(int i=0;i<v.n;i++)
    v.coefs[i]=val;
}

