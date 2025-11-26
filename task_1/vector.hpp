#pragma once // Ensures this header file is only included once per compilation unit

#include <cassert> // For runtime assertions (debug checks)
#include <cstdlib> // For malloc() and free()
#include <cmath>   // For sqrtf()

// Inline helper function to compute the square of a number
static inline double sq(double x)
{
	return x * x;
}

// A simple dynamically allocated vector of doubles
// Provides basic functionality and a cosine similarity operation
struct vector
{
	// Constructor: allocates memory for n elements
	// Requires n to be positive and a multiple of 8 (likely for alignment/SIMD use)
	vector(const int n) : n(n)
	{
		assert(n > 0);								 // Validate size
		data = (double *)malloc(sizeof(double) * n); // Allocate raw memory
		assert(data);								 // Ensure allocation succeeded
	}

	// Destructor: frees allocated memory
	// Invoked automatically when a vector object goes out of scope or is deleted
	~vector()
	{
		free(data);
	}

	// Sets the value of the i-th element in the vector
	void set(double x, int i)
	{
		assert(i >= 0 && i < n); // Bounds check
		data[i] = x;
	}

	// Computes the cosine similarity between two vectors:
	static double cosine_similarity(const vector &a, const vector &b)
	{
		assert(a.n == b.n); // Vectors must be of the same dimension
		const int n = a.n;
		double ab = 0.0; // Dot product accumulator
		double aa = 0.0; // Magnitude accumulator for vector a
		double bb = 0.0; // Magnitude accumulator for vector b

		for (int i = 0; i < n; ++i)
		{
			aa += sq(a.data[i]);
			bb += sq(b.data[i]);
			ab += a.data[i] * b.data[i];
		}

		// Normalize by magnitudes; potential division by zero if aa or bb == 0
		return ab / (sqrtf(aa) * sqrtf(bb));
	}

private:
	const int n = 0;		// Dimension (size) of the vector
	double *data = nullptr; // Pointer to dynamically allocated array
};
