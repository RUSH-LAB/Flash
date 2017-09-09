#include <CL/cl.h>
#include <cmath>

#include "LSHReservoirSampler.h"
#include "dataset.h"
#include "misc.h"
#include "evaluate.h"
#include "indexing.h"
#include "omp.h"
#include "LSHReservoirSampler_config.h"
#include "MatMul.h"
#include "KNN_bruteforce.h"

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include "FrequentItems.h"

using namespace std;

#define NUMHASHBATCH				50
#define BATCHPRINT					5

#define K							4
#define RANGE_POW					15
#define RANGE_ROW_U					15

#define NUMTABLES					32
#define RESERVOIR_SIZE				64
#define OCCUPANCY					1

#define QUERYPROBES					1
#define HASHINGPROBES				1

#define DIMENSION					4000
#define FULL_DIMENSION				16609143
#define NUMBASE						340000
#define MAX_RESERVOIR_RAND			35000
#define NUMQUERY					10000
#define TOPK						128
#define AVAILABLE_TOPK				1024

#define NUMQUERY					10000
#define AVAILABLE_TOPK				1024
#define TOPK						128

#define BASEFILE		"./trigram.svm"
#define QUERYFILE		"./trigram.svm"
#define GTRUTHINDICE	"./webspam_tri_gtruth_indices.txt"
#define GTRUTHDIST		"./webspam_tri_gtruth_distances.txt"

int main(void) {
	
	float etime_0, etime_1, etime_2;
	auto begin = Clock::now();
	auto end = Clock::now();

	std::cout << "Init ... " << std::endl;
	begin = Clock::now();

	LSH *hashFamily = new LSH(2, K, NUMTABLES, RANGE_POW);

	LSHReservoirSampler *myReservoir = new LSHReservoirSampler(hashFamily, RANGE_POW, NUMTABLES, RESERVOIR_SIZE,
		DIMENSION, RANGE_ROW_U, NUMBASE, QUERYPROBES, HASHINGPROBES, OCCUPANCY);
	end = Clock::now();
	etime_0 = (end - begin).count() / 1000000;
	std::cout << "Init Hash and Reservoir Used " << etime_0 << "ms. \n";

	myReservoir->showParams();

#if defined GPU_HASHING
	hashFamily->clLSH(myReservoir->platforms, myReservoir->devices_gpu + CL_GPU_DEVICE, myReservoir->context_gpu,
		myReservoir->program_gpu, myReservoir->command_queue_gpu);
#endif

	int hash_chunk = NUMBASE / NUMHASHBATCH;

	const int nCnt = 10;
	int nList[nCnt] = { 1, 10, 20, 30, 32, 40, 50, 64, 100, TOPK };
	const int gstdCnt = 8;
	float gstdVec[gstdCnt] = { 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.50 };
	const int tstdCnt = 10;
	int tstdVec[tstdCnt] = { 1, 10, 20, 30, 32, 40, 50, 64, 100, TOPK };

	std::cout << "Reading groundtruth ... " << std::endl;
	unsigned int* gtruth_indice = new unsigned int[NUMQUERY * AVAILABLE_TOPK];
	float* gtruth_dist = new float[NUMQUERY * AVAILABLE_TOPK];
	readGroundTruthInt(GTRUTHINDICE, NUMQUERY, AVAILABLE_TOPK, gtruth_indice);
	readGroundTruthFloat(GTRUTHDIST, NUMQUERY, AVAILABLE_TOPK, gtruth_dist);
	similarityOfData(gtruth_dist, NUMQUERY, TOPK, AVAILABLE_TOPK, nList, nCnt);
	std::cout << "Done. \n";

	std::cout << "Reading data ... " << std::endl;
	begin = Clock::now();

	int* sparse_indice = new int[(unsigned)((NUMBASE + NUMQUERY) * DIMENSION)];
	float* sparse_val = new float[(unsigned)((NUMBASE + NUMQUERY) * DIMENSION)];
	int* sparse_marker = new int[(NUMBASE + NUMQUERY) + 1];
	readSparse(BASEFILE, 0, (unsigned)(NUMBASE + NUMQUERY), sparse_indice, sparse_val, sparse_marker, (unsigned)((NUMBASE + NUMQUERY) * DIMENSION));

	end = Clock::now();
	etime_0 = (end - begin).count() / 1000000;
	std::cout << "Reading Used " << etime_0 << "ms. \n";

	std::cout << "Adding ...\n";
	begin = Clock::now();
	for (int b = 0; b < NUMHASHBATCH; b++) {
		myReservoir->add(hash_chunk, sparse_indice, sparse_val, sparse_marker + b * hash_chunk + NUMQUERY);
		if (b % BATCHPRINT == 0) {
			end = Clock::now();
			etime_0 = (end - begin).count() / 1000000;
			std::cout << "Batch " << b << "(vec " << (b * hash_chunk + NUMQUERY) << "), already taken " <<
				etime_0 << " ms." << std::endl;
			myReservoir->checkTableMemLoad();
		}
	}
#if defined GPU_KSELECT || defined GPU_TABLE || defined GPU_HASHING
	clFinish(myReservoir->command_queue_gpu);
	clFinish(myReservoir->command_queue_cpu);
#endif
	end = Clock::now();
	etime_0 = (end - begin).count() / 1000000;
	std::cout << "Adding Used " << etime_0 << "ms. \n";

	myReservoir->checkTableMemLoad();

	std::cout << "Querying...\n";
	unsigned int *queryOutputs = new unsigned int[NUMQUERY * TOPK]();
	begin = Clock::now();
	myReservoir->ann(NUMQUERY, sparse_indice, sparse_val, sparse_marker, queryOutputs, TOPK);

#if defined GPU_KSELECT || defined GPU_TABLE || defined GPU_HASHING /* For accurate timing. */
	clFinish(myReservoir->command_queue_gpu);
	clFinish(myReservoir->command_queue_cpu);
#endif

	end = Clock::now();
	etime_0 = (end - begin).count() / 1000000;
	std::cout << "Querying Used " << etime_0 << "ms. \n";

	similarityMetric(sparse_indice, sparse_val, sparse_marker,
		sparse_indice, sparse_val, sparse_marker + NUMQUERY, queryOutputs, gtruth_dist,
		NUMQUERY, TOPK, AVAILABLE_TOPK, nList, nCnt);

	evaluate( queryOutputs, NUMQUERY, TOPK, gtruth_indice, gtruth_dist, AVAILABLE_TOPK, gstdVec, gstdCnt, tstdVec, tstdCnt, nList, nCnt);				// The number of n interested. 

	delete[] sparse_indice;
	delete[] sparse_val;
	delete[] sparse_marker;
	delete[] gtruth_indice;
	delete[] gtruth_dist;
	delete[] queryOutputs;

	return 0;
}

