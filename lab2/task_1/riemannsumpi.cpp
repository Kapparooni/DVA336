#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

constexpr double correctpi = 3.14159265358979323846264338327950288419716939937510;

inline double f(const double x)
{
	return sqrt(1-x*x);
}

double riemann_seq(const unsigned long n)
{
	const double Dx = 1.0/n;
	double sum = 0.0;
	for(unsigned long i=0; i<n; ++i)
		sum += Dx*f(i*Dx);
	return sum;
}


double riemann_omp(const unsigned long n)
{
	const double Dx = 1.0/n;
	double sum = 0.0;
	
	//omp parallel loop with reduction
	#pragma omp parallel for reduction(+:sum)
	for(unsigned long i=0; i<n; ++i)
		sum += Dx*f(i*Dx);
	
	return sum;
}

int main(int argc, char* argv[]) {
	const unsigned long n = strtoul(argv[1], nullptr, 0);
	const int nproc = omp_get_max_threads();
	assert(n>0 && n%2==0);

	printf("n = %lu\n", n);
	printf("nproc = %d\n", nproc);
	
	double pi;

	double ts = omp_get_wtime();
	pi = 4.0*riemann_seq(n);
	ts = omp_get_wtime()-ts;
	printf("seq, elapsed time = %.3f seconds, err = %.15f\n", ts, abs(correctpi-pi));
	
	//test for omp version
	double t1 = omp_get_wtime();
	pi = 4.0*riemann_omp(n);
	t1 = omp_get_wtime()-t1;
	printf("omp, elapsed time = %.3f seconds, err = %.15f\n", t1, fabs(correctpi-pi));
	
	return EXIT_SUCCESS;
}