/* DO NOT CHANGE THIS FILE */

#pragma once // Ensure this header is included only once

#include <iostream> // For std::ostream
#include <iomanip>	// For std::setprecision
#include <cassert>	// For runtime assertions (not used here but included)

// d-dimensional point representation
// Template parameter 'd' specifies the dimension
template <int d>
struct point
{
	float v[d]; // Coordinates of the point
};

// Overload operator<< to print a point
template <int d>
std::ostream &operator<<(std::ostream &os, const point<d> &p)
{
	os << "(" << p.v[0];		// Print first coordinate
	for (int i = 1; i < d; ++i) // Print remaining coordinates with commas
		os << ',' << p.v[i];
	os << ")"; // Close parentheses
	return os; // Return stream for chaining
}

// d-dimensional ball representation: center point and radius
// Template parameter 'd' specifies the dimension
template <int d>
struct ball
{
	point<d> center; // Center of the ball
	float radius;	 // Radius of the ball
};

// Overload operator<< to print a ball
// Preserves the previous precision of std::cout
template <int d>
std::ostream &operator<<(std::ostream &os, const ball<d> &b)
{
	auto old = std::cout.precision();		  // save current precision to avoid affecting other outputs
	os << "center=" << b.center				  // print center
	   << " radius=" << std::setprecision(10) // set high precision for radius
	   << b.radius							  // print radius with 10 decimals
	   << std::setprecision(old);			  // restore original precision
	return os;
}