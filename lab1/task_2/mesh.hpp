#pragma once

#include "common.hpp"
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

// return square of x
static inline float sq(const float &x)
{
	return x * x;
}

// mesh of n d-dimensional points with vectorized operations
template <const int d>
struct mesh
{
	// constructor: allocate memory for n points
	mesh(const int n) : n(n)
	{
		assert(n > 0 && n % 16 == 0);
		
		// Allocate original AoS layout for compatibility
		data = (point<d> *)malloc(sizeof(point<d>) * n);
		assert(data);
		
		// Allocate SoA layout for vectorized operations
		soa_data = (float **)malloc(sizeof(float *) * d);
		assert(soa_data);
		for (int i = 0; i < d; ++i) {
			soa_data[i] = (float *)malloc(sizeof(float) * n);
			assert(soa_data[i]);
		}
	}

	// destructor: free allocated memory
	~mesh()
	{
		free(data);
		for (int i = 0; i < d; ++i) {
			free(soa_data[i]);
		}
		free(soa_data);
	}

	// set the i-th point in the mesh
	void set(const point<d> p, int i)
	{
		assert(i >= 0 && i < n);
		data[i] = p;
		
		// Also store in SoA layout
		for (int j = 0; j < d; ++j) {
			soa_data[j][i] = p.v[j];
		}
	}

	// compute enclosing ball: center = midpoint of min/max, radius = max distance to center
	ball<d> calc_ball()
	{
		ball<d> b;

		// compute min/max per coordinate to find center (vectorized)
		float min_vals[d], max_vals[d];
		for (int j = 0; j < d; ++j) {
			min_vals[j] = soa_data[j][0];
			max_vals[j] = soa_data[j][0];
		}

		// Process using AVX
		const int simd_width = 8;
		for (int j = 0; j < d; ++j) {
			__m256 min_vec = _mm256_set1_ps(min_vals[j]);
			__m256 max_vec = _mm256_set1_ps(max_vals[j]);
			
			int i = 0;
			for (; i <= n - simd_width; i += simd_width) {
				__m256 data_vec = _mm256_loadu_ps(&soa_data[j][i]);
				min_vec = _mm256_min_ps(min_vec, data_vec);
				max_vec = _mm256_max_ps(max_vec, data_vec);
			}
			
			// Find overall min/max from vector results
			float temp_min[8], temp_max[8];
			_mm256_storeu_ps(temp_min, min_vec);
			_mm256_storeu_ps(temp_max, max_vec);
			
			for (int k = 0; k < 8; ++k) {
				if (temp_min[k] < min_vals[j]) min_vals[j] = temp_min[k];
				if (temp_max[k] > max_vals[j]) max_vals[j] = temp_max[k];
			}
			
			// Handle any remaining elements
			for (; i < n; ++i) {
				float val = soa_data[j][i];
				if (val < min_vals[j]) min_vals[j] = val;
				if (val > max_vals[j]) max_vals[j] = val;
			}
		}

		// center = midpoint of min/max
		for (int i = 0; i < d; ++i) {
			b.center.v[i] = (max_vals[i] - min_vals[i]) * 0.5f + min_vals[i];
		}

		// compute radius = max distance from center (vectorized)
		float maxsqdst = 0.0f;
		const int simd_width_dist = 8;
		
		for (int i = 0; i < n; i += simd_width_dist) {
			__m256 dist_vec = _mm256_setzero_ps();
			
			// Compute squared distance
			for (int j = 0; j < d; ++j) {
				__m256 center_vec = _mm256_set1_ps(b.center.v[j]);
				__m256 data_vec = _mm256_loadu_ps(&soa_data[j][i]);
				__m256 diff_vec = _mm256_sub_ps(data_vec, center_vec);
				__m256 sq_diff_vec = _mm256_mul_ps(diff_vec, diff_vec);
				dist_vec = _mm256_add_ps(dist_vec, sq_diff_vec);
			}
			
			// Update maximum distance
			float distances[8];
			_mm256_storeu_ps(distances, dist_vec);
			for (int k = 0; k < 8 && (i + k) < n; ++k) {
				if (distances[k] > maxsqdst) {
					maxsqdst = distances[k];
				}
			}
		}

		// sqrt only once for efficiency
		b.radius = sqrtf(maxsqdst);

		return b; // return enclosing ball
	}

	// return index of the farthest point from given point p
	int farthest(point<d> p)
	{
		int argmax = 0;
		float maxsqdst = 0.0f;
		const int simd_width = 8;

		// Compute distance for first point
		for (int j = 0; j < d; ++j) {
			float diff = soa_data[j][0] - p.v[j];
			maxsqdst += sq(diff);
		}

		// Search through remaining points
		for (int i = 1; i < n; i += simd_width) {
			__m256 dist_vec = _mm256_setzero_ps();
			
			// Compute squared distances
			for (int j = 0; j < d; ++j) {
				__m256 p_vec = _mm256_set1_ps(p.v[j]);
				__m256 data_vec = _mm256_loadu_ps(&soa_data[j][i]);
				__m256 diff_vec = _mm256_sub_ps(data_vec, p_vec);
				__m256 sq_diff_vec = _mm256_mul_ps(diff_vec, diff_vec);
				dist_vec = _mm256_add_ps(dist_vec, sq_diff_vec);
			}
			
			// Check each distance and update maximum
			float distances[8];
			_mm256_storeu_ps(distances, dist_vec);
			for (int k = 0; k < 8 && (i + k) < n; ++k) {
				if (distances[k] > maxsqdst) {
					maxsqdst = distances[k];
					argmax = i + k;
				}
			}
		}

		return argmax;
	}

private:
	const int n = 0;		   // number of points
	point<d> *data = nullptr;   // AoS layout for compatibility
	float **soa_data = nullptr; // SoA layout for vectorization
};