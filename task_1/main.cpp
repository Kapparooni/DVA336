/* DO NOT CHANGE THIS FILE */

#include "vector.hpp" // Include the vector class definition
#include <ctime>	  // For measuring execution time
#include <iostream>	  // For printing output
#include <cstdlib>	  // For atoi and EXIT_SUCCESS

// Constant to scale rand() output to [0,1)
const double INV_RAND_MAX = 1.0 / RAND_MAX;

int main(int argc, char *argv[])
{
	srand(12345); // Seed the RNG for reproducible results

	// Determine vector size:
	// Use command-line argument if valid, otherwise default
	const int n = (argc < 2 || atoi(argv[1]) < 1) ? 5120000 : atoi(argv[1]);

	vector a(n); // First vector
	vector b(n); // Second vector

	// Fill both vectors with random values in [0,1)
	for (int i = 0; i < n; ++i)
	{
		a.set(rand() * INV_RAND_MAX, i);
		b.set(rand() * INV_RAND_MAX, i);
	}

	clock_t t = clock(); // Start timing

	// Compute cosine similarity between vectors a and b
	float x = vector::cosine_similarity(a, b);

	t = clock() - t; // Compute elapsed CPU time in clock ticks

	// Print results: vector size, similarity, and elapsed time in milliseconds
	std::cout << "n = " << n
			  << "; cosine_similarity = " << x
			  << "; elapsed time = " << 1000.0 * t / CLOCKS_PER_SEC << "ms"
			  << std::endl;

	return EXIT_SUCCESS; // Exit successfully
}
