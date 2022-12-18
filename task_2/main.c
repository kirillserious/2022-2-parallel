#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

const int ROOT = 0;

const double INTEGRAL = log(2) / 2.0 - 0.3125;

// drand returns a double number in boundaries [0, 1.0]
double
drand() {
    return (double)rand()/(double)RAND_MAX;
}

// integrand returns the integrand function
double
integrand() {
	double x = drand();
	double y = drand();
	double z = drand();
	if ((y < 0) || (z < 0) || (x + y + z > 1)) {
		return 0;
	}
	double denominator = 1.0 + x + y + z;
	return 1.0 / (denominator * denominator * denominator);
}

double update_integral(double prev, int prev_count, double add, int add_count) {
	return prev * ((double)prev_count/(double)(prev_count+add_count)) + add / ((double)(prev_count+add_count));
}

int
main(int argc, const char** argv)
{
	int rank, size;
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (argc != 2) {
		if (rank == ROOT) {
			fprintf(stderr, "Precision is not provided\n");
		}
		MPI_Finalize();
		return 0;
	}
	
	double precision;
	if (rank == ROOT) {
		sscanf(argv[1], "%lf", &precision);
	}

	int start_time = time(NULL);
	srand(start_time + 100*rank);

	double integral = 0.0;
	double error;
	int total_parts = 0;

	int next = 1;
	while (next) {
		double part = integrand();
		double part_sum;
		MPI_Reduce(&part, &part_sum, 1, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);
		if (rank == ROOT) {
			integral = update_integral(integral, total_parts, part_sum, size);
			total_parts = total_parts + size;
			if (integral > INTEGRAL) {
				error = integral-INTEGRAL;
			} else {
				error = INTEGRAL-integral;
			}
			if (error < precision) {
				next = 0;
			}
		}
		MPI_Bcast(&next, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
	}

	if (rank == ROOT) {
		printf("Integral=%.10f\n", integral);
		printf("Error=%.10f\n", error);
		printf("PointsCount=%d\n", total_parts);
		int finish_time = time(NULL);
		printf("Time=%d s\n", finish_time-start_time);
	}

	MPI_Finalize();
}
