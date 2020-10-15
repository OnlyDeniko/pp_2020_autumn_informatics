// Copyright 2020 Gruzdeva Diana
#include <vector>
#include "./trapezoidal_integral.h"

double trap_integral(double a, double b, int n, std::function<double(double)> function) {
	double width = (b - a) / n;
	double trapezoidal_integral = 0;
	for (int i = 0; i < n; i++) {
		double x1 = a + i * width;
		double x2 = a + (i + 1) * width;
		trapezoidal_integral += 0.5 * (x2 - x1) * (function(x1) + function(x2));
	}
	return trapezoidal_integral;

}

double call_function(std::function<double(double)> function, double value) {
	return function(value);
}

double func1(double x) {
	return x * x;
}

double func2(double x) {
	return 2 * cos(x) - 5 * x * x * x;
}

double func3(double x) {
	return exp(x * x / 3) * tan(sqrt(x));
}

double func4(double x) {
	return x * x * x + 2 * x * x + 4 * x + 6;
}

double func5(double x) {
	return sin(x) - cos(x);
}
