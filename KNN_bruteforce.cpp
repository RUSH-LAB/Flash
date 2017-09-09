#include "KNN_bruteforce.h"
#include "MatMul.h"
#include "misc.h"

#include <algorithm>
#include <math.h>

struct greater
{
	template<class T>
	bool operator()(T const &a, T const &b) const { return a > b; }
};

/* M vs N (all vectors). */
void KNN_sparse(float *AtA, int *data_indice, float *data_val, int *data_marker_M, int *data_marker_N,
	unsigned int M, unsigned int N) {

	std::cout << "AtA ";
	/* Compute AtA. */
	unsigned int startA, endA, startB, endB;
#pragma omp parallel private(startA, endA, startB, endB)
#pragma omp parallel for
	for (int i = 0; i < M; i++) {
		startA = data_marker_M[i];
		endA = data_marker_M[i + 1];

		for (int j = 0; j < N; j++) { // Versus all. 
			startB = data_marker_N[j];
			endB = data_marker_N[j + 1];
			AtA[(unsigned)(i * N + j)] =
				SparseVecMul(data_indice + startA,
					data_val + startA,
					endA - startA,
					data_indice + startB,
					data_val + startB,
					endB - startB);
		}
	}
	
	std::cout << "normM " << std::endl;
	/* Compute norms. */
	float *M_norm = new float[M];
#pragma omp parallel private(startA, endA, startB, endB)
#pragma omp parallel for
	for (int i = 0; i < M; i++) {
		startA = data_marker_M[i];
		endA = data_marker_M[i + 1];
		M_norm[(unsigned)i] = SparseVecMul(
			data_indice + startA, data_val + startA, endA - startA,
			data_indice + startA, data_val + startA, endA - startA);
	}
	float *N_norm = new float[N];

	std::cout << "normN " << std::endl;
	auto begin = Clock::now();
#pragma omp parallel private(startA, endA, startB, endB)
#pragma omp parallel for
	for (int i = 0; i < N; i++) {
		startA = data_marker_N[i];
		endA = data_marker_N[i + 1];
		N_norm[(unsigned)i] = SparseVecMul(
			data_indice + startA, data_val + startA, endA - startA,
			data_indice + startA, data_val + startA, endA - startA);
	}
	auto end = Clock::now();
	float etime_0 = (end - begin).count() / 1000000;
	std::cout << "normN(repeat) Used " << etime_0 << "ms. \n";

	std::cout << "cosined " << std::endl;
	/* Cosine dists. */
#pragma omp parallel for
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			AtA[(unsigned)(i * N + j)] /= (sqrtf(M_norm[i]) * sqrtf(N_norm[j]));
		}
	}
	std::cout << "sorting" << std::endl;
	/* Sorting. */
#pragma omp parallel for
	for (int i = 0; i < M; i++) {
		std::sort(AtA + N * i, AtA + N * i + N, greater());
	}

	/* Print some of the outputs. */
	//for (int i = 0; i < 10; i++) {
	//	for (int j = 0; j < 10; j++)
	//		std::cout << AtA[i * N + j] << ' ';
	//	std::cout << std::endl << std::endl;
	//}
	//system("pause");

}