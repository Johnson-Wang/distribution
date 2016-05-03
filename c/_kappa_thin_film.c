#include <Python.h>
#define _USE_MATH_DEFINES
#include<math.h>
#include <stdio.h>
#include <numpy/arrayobject.h>
#define THz 1.0e12
#define Angstrom 1.0e-10
#define INVSQRT2PI 0.3989422804014327

static PyObject * py_kappa_thin_film_pvary(PyObject *self, PyObject *args);
static PyObject * py_kappa_thin_film_pconst(PyObject *self, PyObject *args);
static PyObject * py_distribution(PyObject *self, PyObject *args);
static PyObject * py_sumdelta(PyObject *self, PyObject *args);
static PyObject * py_get_thermal_conductivity(PyObject *self, PyObject *args);
static double gaussian(double x, const double sigma)
{
  return INVSQRT2PI / sigma * exp(-x * x / 2 / sigma / sigma);
}  

static void mat_copy_matrix_d3(double a[9], double b[9])
{
  a[0] = b[0];
  a[1] = b[1];
  a[2] = b[2];
  a[3] = b[3];
  a[4] = b[4];
  a[5] = b[5];
  a[6] = b[6];
  a[7] = b[7];
  a[8] = b[8];
}

static void mat_add_vector_d3( double m[3],
			double a[3],
			double b[3] )
{
  int i;
  for ( i = 0; i < 3; i++ ) {
      m[i] = a[i] + b[i];
  }
}

static double mat_get_determinant_d3(double a[9])
{
  return a[0] * (a[4] * a[8] - a[5] * a[7])
    + a[1] * (a[5] * a[6] - a[3] * a[8])
    + a[2] * (a[3] * a[7] - a[4] * a[6]);
}

/* m=axb */
static void mat_multiply_matrix_d3(double m[9],
				   double a[9],
				   double b[9])
{
  int i, j;                   /* a_ij */
  double c[9];
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      c[i*3+j] =
	a[i*3+0] * b[j] + a[i*3+1] * b[1*3+j] + a[i*3+2] * b[2*3+j];
    }
  }
  mat_copy_matrix_d3(m, c);
}

//The tensor is symmetric and thus only the upper triangle is considered
static void mat_vector_outer_product(double v[6],
				     double a[3],
				     double b[3])
{
  v[0] = a[0] * b[0];
  v[1] = a[1] * b[1];
  v[2] = a[2] * b[2];
  v[3] = a[1] * b[2];
  v[4] = a[0] * b[2];
  v[5] = a[0] * b[1];
}

static void mat_add_matrix_d3( double m[9],
			       double a[9],
			       double b[9] )
{
  int i, j;
  for ( i = 0; i < 3; i++ ) {
    for ( j = 0; j < 3; j++ ) {
      m[i*3+j] = a[i*3+j] + b[i*3+j];
    }
  }
}


static void mat_copy_matrix_id3(double a[9], int b[9])
{
  a[0] = b[0];
  a[1] = b[1];
  a[2] = b[2];
  a[3] = b[3];
  a[4] = b[4];
  a[5] = b[5];
  a[6] = b[6];
  a[7] = b[7];
  a[8] = b[8];
}

static void mat_copy_matrix_i3(int a[9], int b[9])
{
  a[0] = b[0];
  a[1] = b[1];
  a[2] = b[2];
  a[3] = b[3];
  a[4] = b[4];
  a[5] = b[5];
  a[6] = b[6];
  a[7] = b[7];
  a[8] = b[8];
}

static void mat_multiply_matrix_vector_d3(double v[3],
				   double a[9],
				   double b[3])
{
  int i;
  double c[3];
  for (i = 0; i < 3; i++)
    c[i] = a[i*3+0] * b[0] + a[i*3+1] * b[1] + a[i*3+2] * b[2];
  for (i = 0; i < 3; i++)
    v[i] = c[i];
}

/* m^-1 */
/* ruby code for auto generating */
/* 3.times {|i| 3.times {|j| */
/*       puts "m[#{j}*3+#{i}]=(a[#{(i+1)%3}*3+#{(j+1)%3}]*a[#{(i+2)%3}*3+#{(j+2)%3}] */
/*	 -a[#{(i+1)%3}*3+#{(j+2)%3}]*a[#{(i+2)%3}*3+#{(j+1)%3}])/det;" */
/* }} */
static void mat_inverse_matrix_d3(double m[9],
				  double a[9])
{
  double det;
  double c[9];
  det = mat_get_determinant_d3(a);

  c[0] = (a[4] * a[8] - a[5] * a[7]) / det;
  c[3] = (a[5] * a[6] - a[3] * a[8]) / det;
  c[6] = (a[3] * a[7] - a[4] * a[6]) / det;
  c[1] = (a[7] * a[2] - a[8] * a[1]) / det;
  c[4] = (a[8] * a[0] - a[6] * a[2]) / det;
  c[7] = (a[6] * a[1] - a[7] * a[0]) / det;
  c[2] = (a[1] * a[5] - a[2] * a[4]) / det;
  c[5] = (a[2] * a[3] - a[0] * a[5]) / det;
  c[8] = (a[0] * a[4] - a[1] * a[3]) / det;
  mat_copy_matrix_d3(m, c);
}

/* m = b^-1 a b */
static void mat_get_similar_matrix_d3(double m[9],
				      double a[9],
				      double b[9])
{
  double c[9];
  mat_inverse_matrix_d3(c, b);
  mat_multiply_matrix_d3(m, a, c);
  mat_multiply_matrix_d3(m, b, m);
}


static void get_delta_summation(const double fc[], const double fa[], double ds[], const int w[], const double s, const int nc, const int nq, const int nb)
{
    double ds_temp;
    int i,j,k;
    for (i=0;i<nc;i++)
      ds[i]=0.;
#pragma omp parallel for private(j,ds_temp, k)
    for (i=0;i<nc;i++)
      for (j=0;j<nq;j++)
      {
	ds_temp=0.;
	for (k=0;k<nb;k++)
	  ds_temp+=gaussian(fa[j*nb+k]-fc[i],s);
	ds[i]+=ds_temp*w[j];
      }
}

static void get_distribution(double c[],
		      const double dist[],
		      const double dest[], 
		      const double sigma, 
		      const int num_q, //number of qpoints
		      const int num_b, //number of phonon branches
		      const int seg)
{
  double d;
  int i, j,k;
#pragma omp parallel for private(d, j, k)
  for (i=0;i<seg; i++){
    d=c[i*2];
    for (j=0;j<num_q;j++)
      for (k=0;k<num_b; k++)
	c[i*2+1]+=dest[j*num_b+k]* gaussian(dist[j*num_b+k]-d,sigma);
  }
}

static void get_kappa_thin_film(const double a,
			 double kthin[],
			 double gamma[],
			 const double iso[], //gamma iso
			 const int nth,  
			 const int nrp, 
			 const int nq, 
			 const int nte,//number of temperature
			 const int nb, 
			 const double thick[],
			 const double rough[],
			 const double qpoints[],
			 const double *spec, //specularity
			 const double cv[],
			 const double gv[], 
			 double cutr,
			 const int dir[],
			 const int dirw){
  int i,j,k,l,m, n;
  double delta, p,factor;
  double  *gv_temp, gv_scalar;
  int alpha,beta;
  alpha=dir[0]; beta=dir[1];
  for (i=0;i<nth*nrp*nq*nte*nb;i++)
    kthin[i]=0.;//initialization
#pragma omp parallel for collapse(5) private(n,delta,p,factor,gv_temp, gv_scalar)
  for (i=0; i<nth;i++){
    for (j=0;j<nrp;j++){
      for (k=0;k<nq;k++){
	for (l=0;l<nte;l++){
	  for (m=0;m<nb;m++){
	    
	    n=i*nrp*nq*nte*nb+j*nq*nte*nb+k*nte*nb+l*nb+m;
	    if (gamma[n]< 0.5 / cutr / THz)
	      continue;
	    gv_temp=gv+k*nb*3+m*3;
	    if (dirw <= 2 && dirw >=0)
	      gv_scalar = fabs(gv_temp[dirw]);
	    else
	      gv_scalar = sqrt(gv_temp[0]*gv_temp[0]+gv_temp[1]*gv_temp[1]+gv_temp[2]*gv_temp[2]);
	    if (gv_scalar<1e-16){
	      factor=1;
	    }
	    else{
	      delta=(2*gamma[n]*2*M_PI*THz) *(thick[i]*a*Angstrom) / (gv_scalar*THz*Angstrom);
 	      if (spec!=NULL){
            p=spec[j];
            factor=delta*(1+p)/ (2*(1-p)+delta*(1+p));
	      }
	      else if (rough!=NULL){
	      	p=exp(-16*M_PI*M_PI*M_PI*rough[j]*rough[j]*qpoints[k*3+2]*qpoints[k*3+2]);
		
		if (fabs(p-1)<1e-5) factor=1;
		else factor=delta*(1+p)/(2*(1-p)+delta*(1+p));
	      }
	    }
	    if (factor>1e-8)
	      gamma[n]/=factor; //boundary scattering effects
	     #pragma omp atomic
	    gamma[n]+=iso[k*nte*nb+l*nb+m]; //isotopic scattering effects
	    kthin[n]=cv[k*nte*nb+l*nb+m]*gv_temp[alpha] * gv_temp[beta] /(2*gamma[n]);
	  }
	}
      }
    }
  }
}

static void get_kappa_at_grid_point(double kappa[],
				    const int kpt_rotations_at_q[],
				    const double rec_lat[], 
				    const double gv[],
				    const double heat_capacity[], 
				    const double scatt[], 
				    const int deg[],
				    const int num_rot,
				    const int num_band,
				    const int num_temp,
				    const int num,
				    const double cutoff_lifetime)
{
  int i,j,k;
  int m,n, remnant;
  double r[9], v[3], v2_tensor[6], v_temp[3];
  double gv2_sum[num_band*6];
  for (i=0;i<num_band*6;i++)
    gv2_sum[i]=0.0;
  for (i=0;i<num_rot;i++)
  {
    mat_copy_matrix_id3(r, kpt_rotations_at_q + i * 9); 
    mat_inverse_matrix_d3(r, r);
    mat_get_similar_matrix_d3(r, r, rec_lat);
    //Considering degenerate group velocity of different branches
    remnant = 0;
    for (j=0;j<num_band;j++)
    {
      if (!(remnant--))
      {
	for (k=0; k<3; k++)
	  v[k]=0;
	for (m=0;m<num_band-j; m++)
	{
	  if (deg[j+m] == j){
	    mat_multiply_matrix_vector_d3(v_temp, r, gv + (j + m) * 3);
	    mat_add_vector_d3(v, v, v_temp);
	  }
	  else
	    break;
	}
	remnant=m-1;
	for (k=0; k<3; k++){
	  v[k] = v[k] / m;
	}
      }
      mat_vector_outer_product(v2_tensor,v,v);
      for (k=0; k<6;k++)
	gv2_sum[j * 6 + k] += v2_tensor[k];
    }
  }
  for (i=0; i<num_temp;i++)
    for (j=0; j<num_band; j++)
    {
      if (scatt[i * num_band + j] < 1.0 / cutoff_lifetime / 1e12)
	continue;
      for (k=0; k<6; k++)
	kappa[i * num_band * 6 + j * 6 + k] = gv2_sum[j * 6 + k] * heat_capacity[i * num_band + j] / scatt[i * num_band + j];
    }
}





static void get_kappa(double kappa[],
		      const int kpt_rotations[],
		      const int mapping[],
		      const double rec_lat[],
		      const double gv[],
		      const double heat_capacity[],
		      const double scatt[],
		      const int degeneracy[],
		      const int num_irreq,
		      const int num_grid,
		      const int num_band,
		      const int num_temp,
		      const double cutoff_lifetime)
{
  int i, j, num_rot, temp_num=0, find;
  int *kpt_rotations_at_q;
  int irr_grid[num_irreq];
  //Find the irreducible grids
  for (i=0; i<num_grid; i++){
    find = 0;
    for (j=0;j<temp_num; j++){
      if (irr_grid[j] == mapping[i]){
	find=1;
	break;
      }
    }
    if (find==0){
      irr_grid[temp_num++]=mapping[i];
    }
  }
  if (temp_num != num_irreq)
  {
    printf("Error! Number of the irreducible grids is wrong!");
    exit(1);
  }

  for (i=0;i<num_irreq; i++){
    num_rot=0;
    for (j=0;j<num_grid; j++)
      if (mapping[j] == irr_grid[i]) num_rot++;
    kpt_rotations_at_q = (int*) malloc(sizeof(int)*num_rot*9);
    num_rot=0;
    for (j=0;j<num_grid;j++)
      if (mapping[j] == irr_grid[i]){
	mat_copy_matrix_i3(kpt_rotations_at_q+num_rot*9, kpt_rotations+j*9);
	num_rot++;
      }
  get_kappa_at_grid_point(kappa + i * num_temp * num_band * 6,
			  kpt_rotations_at_q,
			  rec_lat, 
			  gv + i * num_band * 3,
			  heat_capacity + i * num_temp * num_band, 
			  scatt + i * num_temp * num_band, 
			  degeneracy + i * num_band,
			  num_rot,
			  num_band,
			  num_temp,
			  i,
			  cutoff_lifetime);
  free(kpt_rotations_at_q);
  }
}

static PyMethodDef functions[] = {
  {"kappa_thin_film_pvary", (PyCFunction) py_kappa_thin_film_pvary, METH_VARARGS, "thermal conductivity of thin films for varying p"},
  {"kappa_thin_film_pconst", (PyCFunction) py_kappa_thin_film_pconst, METH_VARARGS, "thermal conductivity of thin film for constant p"},
  {"distribution", (PyCFunction) py_distribution, METH_VARARGS, "distribution of a property in terms of another"},
  {"sumdelta", (PyCFunction) py_sumdelta, METH_VARARGS, "summation of the delta function over the first Brillouin zone "},
  {"thermal_conductivity",py_get_thermal_conductivity, METH_VARARGS, "thermal conductivity calculation" },
  {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC init_kthin(void)
{
  Py_InitModule3("_kthin", functions, "C-extension for kappa_thin_film\n\n...\n");
  return;
}

static PyObject * py_kappa_thin_film_pconst(PyObject *self, PyObject *args)
{
  PyArrayObject* kappa_thin;
  PyArrayObject* py_gamma;
  PyArrayObject* py_gamma_iso;
  PyArrayObject* thick;
  PyArrayObject* specularity;
  PyArrayObject* heat_capacity;
  PyArrayObject* group_velocity;
  PyArrayObject* py_direction;
  double a;
  double cutoff_lifetime;
  int dirw;
  if (!PyArg_ParseTuple(args, "dOOOOOOOdOi",
            &a,
			&kappa_thin,
			&py_gamma,
			&py_gamma_iso,
			&thick,
			&specularity,
			&heat_capacity,
			&group_velocity,
			&cutoff_lifetime,
			&py_direction,
			&dirw))
    return NULL;
  const int num_thick = (int)kappa_thin->dimensions[0];
  const int num_p = (int)kappa_thin->dimensions[1];
  const int num_irreq = (int)kappa_thin->dimensions[2];
  const int num_temp = (int)kappa_thin->dimensions[3];
  const int num_bands = (int)kappa_thin->dimensions[4];
  double *kt=(double*)kappa_thin->data;
  double *gamma=(double*)py_gamma->data;
  const double *gamma_iso=(double*)py_gamma_iso->data;
  const double *th=(double*)thick->data;
  const double *cv=(double*)heat_capacity->data;
  const double *gv=(double*)group_velocity->data;
  const double *p=(double*)specularity->data;
  const int *dir = (int*) py_direction->data;

  get_kappa_thin_film(a,
		      kt,
		      gamma,
		      gamma_iso,
		      num_thick,
		      num_p,
		      num_irreq,
		      num_temp,
		      num_bands,
		      th,
		      NULL,
		      NULL,
		      p,
		      cv,
		      gv,
		      cutoff_lifetime,
		      dir,
		      dirw);
  Py_RETURN_NONE;
}

static PyObject * py_kappa_thin_film_pvary(PyObject *self, PyObject *args)
{
  PyArrayObject* kappa_thin;
  PyArrayObject* py_gamma;
  PyArrayObject* py_gamma_iso;
  PyArrayObject* thick;
  PyArrayObject* rough;
  PyArrayObject* qpoints;
  PyArrayObject* heat_capacity;
  PyArrayObject* group_velocity;
  PyArrayObject* py_direction;
  double a;
  double cutoff_lifetime;
  int dirw;
  if (!PyArg_ParseTuple(args, "dOOOOOOOOdOi",
                        &a,
                        &kappa_thin,
                        &py_gamma,
                        &py_gamma_iso,
                        &thick,
                        &rough,
                        &qpoints,
                        &heat_capacity,
                        &group_velocity,
                        &cutoff_lifetime,
                        &py_direction,
                        &dirw))
    return NULL;
  const int num_thick = (int)kappa_thin->dimensions[0];
  const int num_rough = (int)kappa_thin->dimensions[1];
  const int num_irreq = (int)kappa_thin->dimensions[2];
  const int num_temp = (int)kappa_thin->dimensions[3];
  const int num_bands = (int)kappa_thin->dimensions[4];
  double *kt=(double*)kappa_thin->data;
  double *gamma =(double*)py_gamma->data;
  const double *gamma_iso=(double*)py_gamma_iso->data;
  const double *th=(double*)thick->data;
  const double *r=(double*)rough->data;
  const double *q=(double*)qpoints->data;
  const double *cv=(double*)heat_capacity->data;
  const double *gv=(double*)group_velocity->data;
  const int *dir = (int*) py_direction->data;
  get_kappa_thin_film(a,
		      kt,
		      gamma,
		      gamma_iso,
		      num_thick,
		      num_rough,
		      num_irreq,
		      num_temp,
		      num_bands,
		      th,
		      r,
		      q,
		      NULL,
		      cv,
		      gv,
		      cutoff_lifetime,
		      dir,
		      dirw);
  Py_RETURN_NONE;
}

static PyObject * py_distribution(PyObject *self, PyObject *args)
{
  PyArrayObject* dist;
  PyArrayObject* dest;
  PyArrayObject* py_cast;
  PyArrayObject* kappa;
  double sigma;
  if (!PyArg_ParseTuple(args, "OOOd",
                        &dist,
			&dest,
			&py_cast,
			&sigma))
    return NULL;
  const int nq=(int)dist->dimensions[0];
  const int nb=(int)dist->dimensions[1];
  const int seg=(int)py_cast->dimensions[0];
  const double *di=(double*) dist->data;
  const double *de=(double*)dest->data;
  double *cast=(double*)py_cast->data;
  get_distribution(cast, di, de, sigma , nq, nb, seg);
  Py_RETURN_NONE;
}

static PyObject * py_sumdelta(PyObject *self, PyObject *args)
{
  PyArrayObject* py_fcast;
  PyArrayObject* py_fall;
  PyArrayObject *py_weight;
  PyArrayObject *py_dsum;
  double sigma;
  if (!PyArg_ParseTuple(args, "OOOOd",
		      &py_fcast,
		      &py_fall,
		      &py_dsum,
		      &py_weight,
		      &sigma))
  return NULL;
  const int num_cast=(int)py_fcast->dimensions[0];
  const double * fcast=(double*) py_fcast->data;
  const double * fall=(double*) py_fall->data;
  double* dsum=(double*) py_dsum->data;
  const int *weight = (int*) py_weight->data;
  const int num_q=(int) py_fall->dimensions[0];
  const int num_b=(int) py_fall->dimensions[1];
  get_delta_summation(fcast, fall, dsum, weight,  sigma, num_cast,num_q, num_b);
  Py_RETURN_NONE;
}




static PyObject *py_get_thermal_conductivity(PyObject *self, PyObject *args)
{
  PyArrayObject* py_kappa;
  PyArrayObject* py_scatt;
  PyArrayObject* py_heat_capacity;
  PyArrayObject* py_group_velocity;
  PyArrayObject* py_degeneracy;
  PyArrayObject* py_mapping;
  PyArrayObject* py_kpt_rotations;
  PyArrayObject* py_rec_lat;
  double cutoff_lifetime;

  if (!PyArg_ParseTuple(args, "OOOOOOOOd",
			&py_kappa,
			&py_kpt_rotations,
			&py_mapping,
			&py_rec_lat,
			&py_heat_capacity,
			&py_scatt,
			&py_group_velocity,
			&py_degeneracy,
			&cutoff_lifetime))
    return NULL;
  const int num_irreq = (int)py_kappa->dimensions[0];
  const int num_temp = (int)py_kappa->dimensions[1];
  const int num_band = (int)py_kappa->dimensions[2];
  
  const int num_grid = (int)py_mapping->dimensions[0];
  const int *mapping = (int*)py_mapping->data;
  const double *rec_lat = (double*)py_rec_lat->data;
  const int *kpt_rotations= (int*)py_kpt_rotations->data;
  const double *scatt=(double*)py_scatt->data;
  const double *heat_capacity=(double*)py_heat_capacity->data;
  const double *gv=(double*)py_group_velocity->data;
  const int *degeneracy = (int*) py_degeneracy->data;
  double *kappa= (double*)py_kappa->data;
  get_kappa(kappa,
	    kpt_rotations,
	    mapping,
	    rec_lat,
	    gv,
	    heat_capacity,
	    scatt,
	    degeneracy,
	    num_irreq,
	    num_grid,
	    num_band,
	    num_temp,
	    cutoff_lifetime);
  Py_RETURN_NONE;
}