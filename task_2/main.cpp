/* DO NOT CHANGE THIS FILE */

#include "mesh.hpp" // definition of mesh, point, ball, and related functions
#include <ctime>	// for clock() and measuring CPU time
#include <cstdlib>	// for rand(), srand(), atoi()
#include <iostream> // for std::cout, printing output

// Constant to scale rand() output to [0,2)
const float RAND_SCALE_2 = 2.0f / RAND_MAX;

// generate a random d-dimensional point with coordinates in [0,2)
template <const int d>
inline point<d> randompoint()
{
	point<d> p;
	for (int j = 0; j < d; ++j)
		p.v[j] = rand() * RAND_SCALE_2; // scale rand() to [0,2)
	return p;
}

// D is the dimension, can be set at compile time via -D, default=3
#ifndef D
#define D 3
#endif

int main(int argc, char *argv[])
{
	srand(1234); // fixed seed for reproducibility

	// Determine vector size:
	// Use command-line argument if valid, otherwise default
	const int n = argc < 2 || atoi(argv[1]) < 1 ? 5120000 : atoi(argv[1]);

	mesh<D> m(n); // create mesh with n points

	// fill mesh with random points
	for (int i = 0; i < n; ++i)
		m.set(randompoint<D>(), i);

	// measure time for enclosing ball calculation
	clock_t t = clock();
	ball<D> b = m.calc_ball();
	t = clock() - t;
	std::cout << "The enclosing ball is " << b
			  << "; elapsed time=" << 1000.0 * t / CLOCKS_PER_SEC << "ms" << std::endl;

	// measure time for finding farthest point from center
	t = clock();
	int i = m.farthest(b.center);
	t = clock() - t;
	std::cout << "The farthest point from the ball center is #" << i
			  << "; elapsed time=" << 1000.0 * t / CLOCKS_PER_SEC << "ms" << std::endl;

	return EXIT_SUCCESS; // exit successfully
}
