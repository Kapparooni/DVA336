#pragma once

#include <cassert>
#include <cstdlib>
#include <cmath>

// AVX detection
#ifdef __AVX__
#include <immintrin.h>
#endif

/* Note that for both task1 and task2 I did not see any significant increase 
	in elapsed time (about ~1.2 times faster with AVX, I suspect this is
	because compilation tries to automatically optimize the code?
*/

static inline double sq(double x)
{
	return x * x;
}

struct vector
{
	vector(const int n) : n(n)
	{
		assert(n > 0);
		data = (double *)malloc(sizeof(double) * n);
		assert(data);
	}

	~vector()
	{
		free(data);
	}

	void set(double x, int i)
	{
		assert(i >= 0 && i < n);
		data[i] = x;
	}

	static double cosine_similarity(const vector &a, const vector &b)
	{
		assert(a.n == b.n);
		const int n = a.n;
		double ab = 0.0;
		double aa = 0.0;
		double bb = 0.0;

		#ifdef __AVX__
		// Vectorized part: process 4 elements at a time using AVX
		const int simd_width = 4;
		int i = 0;
		
		__m256d sum_ab_vec = _mm256_setzero_pd();
		__m256d sum_aa_vec = _mm256_setzero_pd();
		__m256d sum_bb_vec = _mm256_setzero_pd();
		
		for (; i <= n - simd_width; i += simd_width) {
			__m256d a_vec = _mm256_loadu_pd(a.data + i);
			__m256d b_vec = _mm256_loadu_pd(b.data + i);
			
			sum_ab_vec = _mm256_add_pd(sum_ab_vec, _mm256_mul_pd(a_vec, b_vec));
			sum_aa_vec = _mm256_add_pd(sum_aa_vec, _mm256_mul_pd(a_vec, a_vec));
			sum_bb_vec = _mm256_add_pd(sum_bb_vec, _mm256_mul_pd(b_vec, b_vec));
		}
		
		// Sum the vector results
		double temp[4];
		_mm256_storeu_pd(temp, sum_ab_vec);
		ab = temp[0] + temp[1] + temp[2] + temp[3];
		_mm256_storeu_pd(temp, sum_aa_vec);
		aa = temp[0] + temp[1] + temp[2] + temp[3];
		_mm256_storeu_pd(temp, sum_bb_vec);
		bb = temp[0] + temp[1] + temp[2] + temp[3];
		
		// Process remaining elements with scalar
		for (; i < n; ++i) {
			aa += sq(a.data[i]);
			bb += sq(b.data[i]);
			ab += a.data[i] * b.data[i];
		}
		#else

		// Original scalar version
		for (int i = 0; i < n; ++i) {
			aa += sq(a.data[i]);
			bb += sq(b.data[i]);
			ab += a.data[i] * b.data[i];
		}
		#endif

		return ab / (sqrt(aa) * sqrt(bb));
	}

private:
	const int n = 0;
	double *data = nullptr;
};