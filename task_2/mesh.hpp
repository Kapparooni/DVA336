#pragma once // Ensure this header is included only once

#include "common.hpp" // definitions of point, ball, etc.
#include <cassert>	  // for assert()
#include <cstdlib>	  // for malloc() and free()
#include <cmath>	  // for sqrtf()

// return square of x
static inline float sq(const float &x)
{
	return x * x;
}

// return squared distance between two d-dimensional points
template <int d>
static inline float sqdst(const point<d> &a, const point<d> &b)
{
	float sum = 0.0f;
	for (int i = 0; i < d; ++i)
		sum += sq(a.v[i] - b.v[i]);
	return sum;
}

// mesh of n d-dimensional points (AoS layout)
template <const int d>
struct mesh
{
	// constructor: allocate memory for n points
	mesh(const int n) : n(n)
	{
		assert(n > 0 && n % 16 == 0);					 // size must be positive and multiple of 16
		data = (point<d> *)malloc(sizeof(point<d>) * n); // allocate memory
		assert(data);
	}

	// destructor: free allocated memory
	~mesh()
	{
		free(data);
	}

	// set the i-th point in the mesh
	void set(const point<d> p, int i)
	{
		assert(i >= 0 && i < n);
		data[i] = p;
	}

	// compute enclosing ball: center = midpoint of min/max, radius = max distance to center
	ball<d> calc_ball()
	{
		ball<d> b;

		// compute min/max per coordinate to find center
		point<d> min = data[0];
		point<d> max = data[0];
		for (int i = 1; i < n; ++i)
			for (int j = 0; j < d; ++j)
				min.v[j] = data[i].v[j] < min.v[j] ? data[i].v[j] : min.v[j],
				max.v[j] = data[i].v[j] > max.v[j] ? data[i].v[j] : max.v[j];

		// center = midpoint of min/max
		for (int i = 0; i < d; ++i)
			b.center.v[i] = (max.v[i] - min.v[i]) * 0.5f + min.v[i];

		// compute radius = max distance from center
		float tmp, maxsqdst = 0.0f;
		for (int i = 0; i < n; ++i)
		{
			tmp = sqdst(data[i], b.center);
			maxsqdst = maxsqdst > tmp ? maxsqdst : tmp;
		}
		// sqrt only once for efficiency
		b.radius = sqrtf(maxsqdst);

		return b; // return enclosing ball
	}

	// return index of the farthest point from given point p
	int farthest(point<d> p)
	{
		int argmax = 0;
		float maxsqdst = sqdst(data[0], p);
		for (int i = 1; i < n; ++i)
		{
			float sqdsti = sqdst(data[i], p);
			if (sqdsti > maxsqdst)
			{
				maxsqdst = sqdsti;
				argmax = i;
			}
		}
		return argmax;
	}

private:
	const int n = 0;		  // number of points
	point<d> *data = nullptr; // pointer to points
};
