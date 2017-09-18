
#include <cmath>

#include "LSHReservoirSampler.h"
#include "dataset.h"
#include "misc.h"
#include "evaluate.h"
#include "indexing.h"
#include "omp.h"
#include "benchmarking.h"
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

/** This function benchmarks the k-selection on GPU, CPU etc. 
*/
void benchmark_kselect() {
	float etime_0, etime_1, etime_2;
	auto begin = Clock::now();
	auto end = Clock::now();

	/* Parameters zoo. */
	const int numL = 6;
	int Ls[numL] = { 16,32,64,128,256,512 };
	const int numK = 1;
	int Ks[numK] = { 4 };
	const int numRangePows = 1;
	int rangePows[numRangePows] = { 15 };
	const int numResSize = 1;
	int resSizes[numResSize] = { 32 };
	const int numQProbes = 1;
	int Qprobes[numQProbes] = { 8 };
	const int numHProbes = 1;
	int Hprobes[numHProbes] = { 1 };
	const int numResFracs = 1;
	float ResFracs[numResFracs] = { 1 };

	std::cout << "Reading groundtruth ... " << std::endl;
	unsigned int* gtruth_indice = new unsigned int[NUMQUERY * AVAILABLE_TOPK];
	float* gtruth_dist = new float[NUMQUERY * AVAILABLE_TOPK];
	readGroundTruthInt(GTRUTHINDICE, NUMQUERY, AVAILABLE_TOPK, gtruth_indice);
	readGroundTruthFloat(GTRUTHDIST, NUMQUERY, AVAILABLE_TOPK, gtruth_dist);
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
	unsigned int *queryOutputs = new unsigned int[NUMQUERY * TOPK]();
	int hash_chunk = NUMBASE / NUMHASHBATCH;
	unsigned int *ann_out = new unsigned int[TOPK * hash_chunk]();

	/* Dummy. */
	LSH *hashFamily = new LSH(2, K, NUMTABLES, RANGE_POW);
	LSHReservoirSampler *myReservoir = new LSHReservoirSampler(hashFamily, RANGE_POW, NUMTABLES, RESERVOIR_SIZE,
		DIMENSION, RANGE_ROW_U, NUMBASE, QUERYPROBES, HASHINGPROBES, 0.01);

	float *timings = new float[4];
	float *gpu_clever = new float[numL]();
	float *gpu_naive = new float[numL]();
	float *cpu_multi = new float[numL]();
	float *cpu_single = new float[numL]();
	int *segmnt = new int[numL]();

	/* Grid search. */
	for (int h = 0; h < numHProbes; h++) {
		for (int rs = 0; rs < numResSize; rs++) {
			for (int k = 0; k < numK; k++) {
				for (int l = 0; l < numL; l++) {
					for (int rp = 0; rp < numRangePows; rp++) {
						for (int f = 0; f < numResFracs; f++) {
							for (int q = 0; q < numQProbes; q++) {

								std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl << std::endl;

								std::cout << "HProbe " << Hprobes[h] << std::endl;
								std::cout << "Qprobe " << Qprobes[q] << std::endl;
								std::cout << "K " << Ks[k] << std::endl;
								std::cout << "L " << Ls[l] << std::endl;
								std::cout << "rangePow " << rangePows[rp] << std::endl;
								std::cout << "resSize " << resSizes[rs] << std::endl;
								std::cout << "ResFrac " << ResFracs[f] << std::endl << std::endl;

								std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl << std::endl;

								//std::cout << "Init ... " << std::endl;
								//begin = Clock::now();
								LSH *hashFamily = new LSH(2, Ks[k], Ls[l], rangePows[rp]);
								myReservoir->restart(hashFamily, rangePows[rp], Ls[l], resSizes[rs],
									DIMENSION, rangePows[rp], NUMBASE, Qprobes[q], Hprobes[h], ResFracs[f]);
								end = Clock::now();
								//etime_0 = (end - begin).count() / 1000000;
								//std::cout << "Init Hash and Reservoir Used " << etime_0 << "ms. \n";
								//myReservoir->showParams();

								//std::cout << "Adding ...\n";
								//begin = Clock::now();
								for (int b = 0; b < NUMHASHBATCH; b++) {
									myReservoir->add(hash_chunk, sparse_indice, sparse_val, sparse_marker + b * hash_chunk + NUMQUERY);
									//if (b % BATCHPRINT == 0) {
									//	end = Clock::now();
									//	etime_0 = (end - begin).count() / 1000000;
									//	std::cout << "Batch " << b << "(vec " << (b * hash_chunk + NUMQUERY) << "), already taken " <<
									//		etime_0 << " ms." << std::endl;
									//	myReservoir->checkTableMemLoad();
									//}
								}
								//end = Clock::now();
								//etime_0 = (end - begin).count() / 1000000;
								//std::cout << "Adding Used " << etime_0 << "ms. \n";

								int retests = 1;

								for (int b = 0; b < retests; b++) {
									//segmnt[l] = myReservoir->benchCounting(10000, sparse_indice, sparse_val, sparse_marker + b * 10000, timings);
									gpu_clever[l] += timings[0];
									gpu_naive[l] += timings[1];
									cpu_multi[l] += timings[2];
									cpu_single[l] += timings[3];
								}

								gpu_clever[l] /= retests;
								gpu_naive[l] /= retests;
								cpu_multi[l] /= retests;
								cpu_single[l] /= retests;

							}
						}
					}
				}
			}
		}
	}

	std::cout << ">>>>>>>>>>>> SegmentSize" << std::endl;
	for (int i = 0; i < numL; i++) std::cout << segmnt[i] << ' ';
	std::cout << std::endl;

	std::cout << ">>>>>>>>>>>> GPU CountReduce" << std::endl;
	for (int i = 0; i < numL; i++) std::cout << gpu_clever[i] << ' ';
	std::cout << std::endl;

	std::cout << ">>>>>>>>>>>> GPU Naive" << std::endl;
	for (int i = 0; i < numL; i++) std::cout << gpu_naive[i] << ' ';
	std::cout << std::endl;

	std::cout << ">>>>>>>>>>>> CPU MultiThread" << std::endl;
	for (int i = 0; i < numL; i++) std::cout << cpu_multi[i] << ' ';
	std::cout << std::endl;

	std::cout << ">>>>>>>>>>>> GPU SingleThread" << std::endl;
	for (int i = 0; i < numL; i++) std::cout << cpu_single[i] << ' ';
	std::cout << std::endl;

#ifdef VISUAL_STUDIO
	system("pause");
#endif	
	delete[] gpu_clever;
	delete[] gpu_naive;
	delete[] cpu_multi;
	delete[] cpu_single;

	delete[] timings;
	delete[] ann_out;
	delete[] sparse_indice;
	delete[] sparse_val;
	delete[] sparse_marker;
	delete[] gtruth_indice;
	delete[] gtruth_dist;
	delete[] queryOutputs;

}



/** This function benchmarks the naive random projection. 

@param RANDPROJ_COMPRESS The number of random projections to generate. 
*/
void benchmark_naiverp(int RANDPROJ_COMPRESS) {
	float etime_0, etime_1, etime_2;
	auto begin = Clock::now();
	auto end = Clock::now();

	int numHashesToGen = RANDPROJ_COMPRESS;
	int numVecs = NUMBASE + NUMQUERY;

	int* sparse_indice = new int[(unsigned)(numVecs * DIMENSION)];
	float* sparse_val = new float[(unsigned)(numVecs * DIMENSION)];
	int* sparse_marker = new int[numVecs + 1];
	readSparse(BASEFILE, 0, (unsigned)numVecs, sparse_indice, sparse_val, sparse_marker, (unsigned)(numVecs * DIMENSION));

	/* Generate random numbers. */
	std::cout << "Generating random numbers for random projection. " << std::endl;
	begin = Clock::now();
	unsigned int numRandToGen = numHashesToGen * FULL_DIMENSION;
	float *myRands = new float[numRandToGen];
#pragma omp parallel for
	for (int i = 0; i < numRandToGen; i++) {
		myRands[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}
	end = Clock::now();
	etime_0 = (end - begin).count() / 1000000;
	std::cout << "Used " << etime_0 << "ms. \n";
	std::cout << "Completed generating random numbers for random projection. " << std::endl;

	std::cout << "Generating random projection. " << std::endl;
	begin = Clock::now();
	int hash_chunk = NUMBASE / NUMHASHBATCH;
	unsigned int numOutputsToGen = numHashesToGen * numVecs;
	float *outputs = new float[numOutputsToGen];
	size_t startVec, endVec;
	for (int b = 0; b < NUMHASHBATCH; b++) {
		if (b % BATCHPRINT == 0) {
			end = Clock::now();
			etime_0 = (end - begin).count() / 1000000;
			std::cout << "Batch " << b << "(vec " << (b * hash_chunk + NUMQUERY) << "), already taken " <<
				etime_0 << " ms." << std::endl;
		}
#pragma omp parallel private(startVec, endVec)
#pragma omp parallel for
		for (int i = 0; i < hash_chunk; i++) {
			startVec = sparse_marker[hash_chunk * b + i];
			endVec = sparse_marker[hash_chunk * b + i + 1];
			for (int h = 0; h < numHashesToGen; h++) {
				outputs[(hash_chunk * b + i) * numHashesToGen + h] =
					SparseVecMul(sparse_indice + startVec, sparse_val + startVec, endVec - startVec, myRands + h * FULL_DIMENSION);
			}
		}
	}
	end = Clock::now();
	std::cout << "Used " << etime_0 << "ms. \n";
	std::cout << "Completed generating random projections. " << std::endl;

	std::cout << "Saving to file ... " << std::endl;
	std::ofstream outfile(BASEFILE + (string)".RP");
	for (int i = 0; i < numVecs; i++) {
		for (int h = 0; h < numHashesToGen; h++) {
			outfile << outputs[i * numHashesToGen + h] << ' ';
		}
		outfile << std::endl;
	}
	outfile.close();

}


/** This function performs grid search over the parameter space. 
*/
void benchmark_paragrid() {
	float etime_0, etime_1, etime_2;
	auto begin = Clock::now();
	auto end = Clock::now();

	/* Parameters zoo. */
	const int numL = 20;
	int Ls[numL] = { 10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200 };
	const int numK = 1;
	int Ks[numK] = { 4 };
	const int numRangePows = 1;
	int rangePows[numRangePows] = { 15 };
	const int numResSize = 1;
	int resSizes[numResSize] = { 32 };
	const int numQProbes = 1; // For repetition only. 
	int Qprobes[numQProbes] = { 1 };
	const int numHProbes = 1;
	int Hprobes[numHProbes] = { 1 };
	const int numResFracs = 1;
	float ResFracs[numResFracs] = { 1 };

	/* Metrics zoo. */
	const int nCnt = 10;
	int nList[nCnt] = { 1, 10, 20, 30, 32, 40, 50, 64, 100, TOPK };
	const int gstdCnt = 8;
	float gstdVec[gstdCnt] = { 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.50 };
	const int tstdCnt = 10;
	int tstdVec[tstdCnt] = { 1, 10, 20, 30, 32, 40, 50, 64, 100, TOPK };

	std::cout << "Reading groundtruth ... Groundtruth topk similarities ..." << std::endl;
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
	unsigned int *queryOutputs = new unsigned int[NUMQUERY * TOPK]();
	int hash_chunk = NUMBASE / NUMHASHBATCH;
	unsigned int *ann_out = new unsigned int[TOPK * hash_chunk]();

	/* Dummy. */
	LSH *hashFamily = new LSH(2, K, NUMTABLES, RANGE_POW);
	LSHReservoirSampler *myReservoir = new LSHReservoirSampler(hashFamily, RANGE_POW, NUMTABLES, RESERVOIR_SIZE,
		DIMENSION, RANGE_ROW_U, NUMBASE, QUERYPROBES, HASHINGPROBES, 0.01);

	/* Grid search. */
	for (int h = 0; h < numHProbes; h++) {
		for (int rs = 0; rs < numResSize; rs++) {
			for (int k = 0; k < numK; k++) {
				for (int l = 0; l < numL; l++) {
					for (int rp = 0; rp < numRangePows; rp++) {
						for (int f = 0; f < numResFracs; f++) {
							for (int q = 0; q < numQProbes; q++) {

								std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl << std::endl;

								std::cout << "HProbe " << Hprobes[h] << std::endl;
								std::cout << "Qprobe " << Qprobes[q] << std::endl;
								std::cout << "K " << Ks[k] << std::endl;
								std::cout << "L " << Ls[l] << std::endl;
								std::cout << "rangePow " << rangePows[rp] << std::endl;
								std::cout << "resSize " << resSizes[rs] << std::endl;
								std::cout << "ResFrac " << ResFracs[f] << std::endl << std::endl;

								std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl << std::endl;

								std::cout << "Init ... " << std::endl;
								begin = Clock::now();
								LSH *hashFamily = new LSH(2, Ks[k], Ls[l], rangePows[rp]);
								myReservoir->restart(hashFamily, rangePows[rp], Ls[l], resSizes[rs],
									DIMENSION, rangePows[rp], NUMBASE, Qprobes[q], Hprobes[h], ResFracs[f]);
								end = Clock::now();
								etime_0 = (end - begin).count() / 1000000;
								std::cout << "Init Hash and Reservoir Used " << etime_0 << "ms. \n";
								myReservoir->showParams();

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
								end = Clock::now();
								etime_0 = (end - begin).count() / 1000000;
								std::cout << "Adding Used " << etime_0 << "ms. \n";

								/* TODO. Note that here the adding did not take into account the query
								vectors, which causes the adding timing to be slightly shorter than
								all versus all adding time. However, since the number of query
								vectors are so small in comparison to the number of all vectors, the
								timing difference is ignored. */

								myReservoir->checkTableMemLoad();

#if defined PARAM_GRID_AVA_TIMING
								std::cout << "ANN ... " << std::endl;
								begin = Clock::now();
								for (int b = 0; b < NUMHASHBATCH; b++) {
									if (b % BATCHPRINT == 0) {
										end = Clock::now();
										etime_0 = (end - begin).count() / 1000000;
										std::cout << "Batch " << b << "(vec " << (b * hash_chunk) << "), already taken " <<
											etime_0 << " ms." << std::endl;
									}
									myReservoir->ann(hash_chunk, sparse_indice, sparse_val, sparse_marker + hash_chunk * b,
										ann_out, TOPK);
								}
								end = Clock::now();
								etime_0 = (end - begin).count() / 1000000;
								std::cout << "ANN Used " << etime_0 << "ms. \n";
#endif
								std::cout << "Quality Eval...\n";
								begin = Clock::now();
								myReservoir->ann(NUMQUERY, sparse_indice, sparse_val, sparse_marker, queryOutputs, TOPK);
								end = Clock::now();
								etime_0 = (end - begin).count() / 1000000;
								std::cout << "Querying eval data Used " << etime_0 << "ms. \n";

								similarityMetric(sparse_indice, sparse_val, sparse_marker,
									sparse_indice, sparse_val, sparse_marker + NUMQUERY, queryOutputs, gtruth_dist,
									NUMQUERY, TOPK, AVAILABLE_TOPK, nList, nCnt);

								evaluate(queryOutputs, NUMQUERY, TOPK, gtruth_indice, gtruth_dist, AVAILABLE_TOPK,
									gstdVec, gstdCnt, tstdVec, tstdCnt, nList, nCnt);
							}
						}
					}
				}
			}
		}
	}

#ifdef VISUAL_STUDIO
	system("pause");
#endif	
	delete[] ann_out;
	delete[] sparse_indice;
	delete[] sparse_val;
	delete[] sparse_marker;
	delete[] gtruth_indice;
	delete[] gtruth_dist;
	delete[] queryOutputs;
}

void benchmark_bruteforce() {
	float etime_0, etime_1, etime_2;
	auto begin = Clock::now();
	auto end = Clock::now();
	unsigned int num_vectors = NUMBASE + NUMQUERY;
	int* data_indice = new int[(unsigned)(num_vectors * DIMENSION)];
	float* data_val = new float[(unsigned)(num_vectors * DIMENSION)];
	int* data_marker = new int[num_vectors + 1];
	int chunk = num_vectors / NUMHASHBATCH;
	float *AtA = new float[(unsigned int)(chunk * num_vectors)];

	std::cout << "Reading data ... " << std::endl;
	etime_0 = 0;
	begin = Clock::now();
	readSparse(BASEFILE, 0, (unsigned)(num_vectors), data_indice, data_val, data_marker, (unsigned)(num_vectors * DIMENSION));

	end = Clock::now();
	etime_0 += (end - begin).count() / 1000000;
	std::cout << "Reading Used " << etime_0 << "ms. \n";

	std::cout << "Bruteforce ... " << std::endl;
	begin = Clock::now();
	for (int i = 0; i < NUMHASHBATCH; i++) {
		if (i % BATCHPRINT == 0) {
			end = Clock::now();
			etime_0 = (end - begin).count() / 1000000;
			std::cout << "Batch " << i << "(vec " << (i * chunk) << "), already taken " <<
				etime_0 << " ms." << std::endl;
		}
		KNN_sparse(AtA, data_indice, data_val, data_marker + i * chunk,
			data_marker, chunk, num_vectors);
	}
	end = Clock::now();
	etime_0 = (end - begin).count() / 1000000;
	std::cout << "Bruteforce Used " << etime_0 << "ms. \n";

	delete[] AtA;
	delete[] data_indice;
	delete[] data_marker;
	delete[] data_val;
}

/** This function performs all versus all knn graph, and evaluate the quality. 
*/
void benchmark_ava() {
	float etime_0, etime_1, etime_2;
	auto begin = Clock::now();
	auto end = Clock::now();

	unsigned int num_vectors = NUMBASE + NUMQUERY;

	std::cout << "Init ... " << std::endl;
	begin = Clock::now();
	LSH *hashFamily = new LSH(2, K, NUMTABLES, RANGE_POW);
	LSHReservoirSampler *myReservoir = new LSHReservoirSampler(hashFamily, RANGE_POW, NUMTABLES, RESERVOIR_SIZE,
		DIMENSION, RANGE_ROW_U, num_vectors, QUERYPROBES, HASHINGPROBES, OCCUPANCY);
	end = Clock::now();
	etime_0 = (end - begin).count() / 1000000;
	std::cout << "Init Hash and Reservoir Used " << etime_0 << "ms. \n";

	myReservoir->showParams();

	int* data_indice = new int[(unsigned)(num_vectors * DIMENSION)];
	float* data_val = new float[(unsigned)(num_vectors * DIMENSION)];
	int* data_marker = new int[num_vectors + 1];

	std::cout << "Reading data ... " << std::endl;
	etime_0 = 0;
	begin = Clock::now();
	readSparse(BASEFILE, 0, (unsigned)(num_vectors), data_indice, data_val, data_marker, (unsigned)(num_vectors * DIMENSION));

	end = Clock::now();
	etime_0 += (end - begin).count() / 1000000;
	std::cout << "Reading Used " << etime_0 << "ms. \n";

	unsigned int *ann_out = new unsigned int[TOPK * num_vectors];

	/* All versus all hashing and querying. */

	int chunk = num_vectors / NUMHASHBATCH;
	unsigned int startA, endA, startB, endB;

	std::cout << "Adding ... " << std::endl;
	begin = Clock::now();
	for (int i = 0; i < NUMHASHBATCH; i++) {
		if (i % BATCHPRINT == 0) {
			end = Clock::now();
			etime_0 = (end - begin).count() / 1000000;
			std::cout << "Batch " << i << "(vec " << (i * chunk) << "), already taken " <<
				etime_0 << " ms." << std::endl;
			myReservoir->checkTableMemLoad();
		}
		myReservoir->add(chunk, data_indice, data_val, data_marker + chunk * i);
	}
	end = Clock::now();
	etime_0 = (end - begin).count() / 1000000;
	std::cout << "Adding Used " << etime_0 << "ms. \n";

	myReservoir->checkTableMemLoad();

	std::cout << "ANN ... " << std::endl;
	begin = Clock::now();
	for (int i = 0; i < NUMHASHBATCH; i++) {
		if (i % BATCHPRINT == 0) {
			end = Clock::now();
			etime_0 = (end - begin).count() / 1000000;
			std::cout << "Batch " << i << "(vec " << (i * chunk) << "), already taken " <<
				etime_0 << " ms." << std::endl;
		}
		myReservoir->ann(chunk, data_indice, data_val, data_marker + chunk * i,
			ann_out, TOPK); // + chunk * i * TOPK
	}
	end = Clock::now();
	etime_0 = (end - begin).count() / 1000000;
	std::cout << "ANN Used " << etime_0 << "ms. \n";

#if defined MATMUL

	std::cout << "Reading groundtruth ..." << std::endl;
	unsigned int* gtruth_indice = new unsigned int[NUMQUERY * AVAILABLE_TOPK];
	readGroundTruthInt(GTRUTHINDICE, NUMQUERY, AVAILABLE_TOPK, gtruth_indice);
	std::cout << "Done. \n";

	std::cout << "Approx AtA ... " << std::endl;
	begin = Clock::now();
	float *approxAtA = new float[(unsigned int)(NUMQUERY * num_vectors)]();
#pragma omp parallel private(startA, endA, startB, endB)
#pragma omp parallel for
	for (int i = 0; i < NUMQUERY; i++) { // Only multiply and evaluate queries. 
		startA = data_marker[i];
		endA = data_marker[i + 1];
		for (int j = 0; j < TOPK; j++) {
			unsigned int output_idx = min(num_vectors - 1, ann_out[i * TOPK + j]);
			startB = data_marker[output_idx];
			endB = data_marker[output_idx + 1];
			approxAtA[(unsigned)(i * num_vectors + output_idx)] =
				SparseVecMul(data_indice + startA,
					data_val + startA,
					endA - startA,
					data_indice + startB,
					data_val + startB,
					endB - startB);
		}
	}
	end = Clock::now();
	etime_0 = (end - begin).count() / 1000000;
	std::cout << "Approx AtA Used " << etime_0 << "ms. Computed " << NUMQUERY << " rows. \n";

	std::cout << "Evaluate Approx. AtA ... " << std::endl;
	float *AtA = new float[(unsigned int)(NUMQUERY * num_vectors)]();
#pragma omp parallel private(startA, endA, startB, endB)
#pragma omp parallel for
	for (int i = 0; i < NUMQUERY; i++) {
		startA = data_marker[i];
		endA = data_marker[i + 1];

		/* Self. */
		AtA[(unsigned)(i * num_vectors + i)] =
			SparseVecMul(data_indice + startA,
				data_val + startA,
				endA - startA,
				data_indice + startA,
				data_val + startA,
				endA - startA);

		for (int j = 0; j < (TOPK - 1); j++) {
			startB = data_marker[gtruth_indice[i * AVAILABLE_TOPK + j]];
			endB = data_marker[gtruth_indice[i * AVAILABLE_TOPK + j] + 1];
			AtA[(unsigned)(i * num_vectors + gtruth_indice[i * AVAILABLE_TOPK + j])] =
				SparseVecMul(data_indice + startA,
					data_val + startA,
					endA - startA,
					data_indice + startB,
					data_val + startB,
					endB - startB);
		}
	}

	float total_l1 = 0;
	float total_l2 = 0;

	for (int i = 0; i < (NUMQUERY * num_vectors); i++) {
		//if (AtA[i] - approxAtA[i] > 0) std::cout << (AtA[i] - approxAtA[i]) << ' ';
		total_l1 += abs(AtA[i] - approxAtA[i]);
		total_l2 += pow(AtA[i] - approxAtA[i], 2);
	}

	end = Clock::now();
	etime_0 = (end - begin).count() / 1000000;
	std::cout << "Eval Approx AtA Used " << etime_0 << "ms. \n";
	std::cout << "L1 distance " << total_l1 << " for " << NUMQUERY << " rows\n";
	std::cout << "Squared L2 distance " << total_l2 << " for " << NUMQUERY << " rows\n";
	delete[] gtruth_indice;
	delete[] approxAtA;
	delete[] AtA;

#endif // MATMUL. 

	delete[] data_indice;
	delete[] data_marker;
	delete[] data_val;
	delete[] ann_out;

}

void benchmark_friendster_quality() {
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
		myReservoir->add(hash_chunk, sparse_indice, sparse_val, sparse_marker + b * hash_chunk);
		if (b % BATCHPRINT == 0) {
			end = Clock::now();
			etime_0 = (end - begin).count() / 1000000;
			std::cout << "Batch " << b << "(vec " << (b * hash_chunk) << "), already taken " <<
				etime_0 << " ms." << std::endl;
			myReservoir->checkTableMemLoad();
		}
	}
	end = Clock::now();
	etime_0 = (end - begin).count() / 1000000;
	std::cout << "Adding Used " << etime_0 << "ms. \n";

	myReservoir->checkTableMemLoad();

	std::cout << "Querying...\n";
	unsigned int *queryOutputs = new unsigned int[NUMQUERY * TOPK];
	begin = Clock::now();
	myReservoir->ann(NUMQUERY, sparse_indice, sparse_val, sparse_marker + NUMBASE, queryOutputs, TOPK);
	end = Clock::now();
	etime_0 = (end - begin).count() / 1000000;
	std::cout << "Querying Used " << etime_0 << "ms. \n";

	similarityMetric(sparse_indice, sparse_val, sparse_marker + NUMBASE,
		sparse_indice, sparse_val, sparse_marker, queryOutputs, gtruth_dist,
		NUMQUERY, TOPK, AVAILABLE_TOPK, nList, nCnt);

	evaluate(
		queryOutputs,		// The output indices of queries. 
		NUMQUERY,		// The number of query entries, should be the same for outputs and groundtruths. 
		TOPK,				// The topk per query contained in the queryOutputs. 
		gtruth_indice,		// The groundtruth indice vector. 
		gtruth_dist,			// The groundtruth distance vector. 
		AVAILABLE_TOPK,	// Available topk information in the groundtruth. 
		gstdVec,			// The goldstandards (similarity to be tested, a vector). 
		gstdCnt,			// The number of goldstandards. 
		tstdVec,			// The Tstandards (top k gtruth to be tested, a vector). 
		tstdCnt,			// The number of Tstandards.
		nList,				// The R@n or G@n interested, a vector. 
		nCnt);				// The number of n interested. 

#ifdef VISUAL_STUDIO
	system("pause");
#endif	

	delete[] sparse_indice;
	delete[] sparse_val;
	delete[] sparse_marker;
	delete[] gtruth_indice;
	delete[] gtruth_dist;
	delete[] queryOutputs;
}

void benchmark_sparse() {
	float etime_0, etime_1, etime_2;
	auto begin = Clock::now();
	auto end = Clock::now();

	std::cout << "Initializing data structures and random numbers ..." << std::endl;
	begin = Clock::now();
	LSH *hashFamily = new LSH(2, K, NUMTABLES, RANGE_POW); // Initialize LSH hash. 
	LSHReservoirSampler *myReservoir = new LSHReservoirSampler(hashFamily, RANGE_POW, NUMTABLES, RESERVOIR_SIZE,
		DIMENSION, RANGE_ROW_U, NUMBASE, QUERYPROBES, HASHINGPROBES, OCCUPANCY); // Initialize hashtables and other datastructures. 
#if defined OPENCL_HASHING
	/* If using GPU hashing, initialize GPU environment for hashfunction. */
	hashFamily->clLSH(myReservoir->platforms, myReservoir->devices_gpu + CL_DEVICE_ID, myReservoir->context_gpu,
		myReservoir->program_gpu, myReservoir->command_queue_gpu);
#endif
	end = Clock::now();
	etime_0 = (end - begin).count() / 1000000;
	std::cout << "Completed, used " << etime_0 << "ms. \n";

	myReservoir->showParams(); // Print out parameters. 

	std::cout << "Reading groundtruth ... " << std::endl;
	unsigned int* gtruth_indice = new unsigned int[NUMQUERY * AVAILABLE_TOPK];
	float* gtruth_dist = new float[NUMQUERY * AVAILABLE_TOPK];
	readGroundTruthInt(GTRUTHINDICE, NUMQUERY, AVAILABLE_TOPK, gtruth_indice);
	readGroundTruthFloat(GTRUTHDIST, NUMQUERY, AVAILABLE_TOPK, gtruth_dist);
	std::cout << "Completed. \n";

	std::cout << "Reading data ... " << std::endl;
	begin = Clock::now();
	int* sparse_indice = new int[(unsigned)((NUMBASE + NUMQUERY) * DIMENSION)];
	float* sparse_val = new float[(unsigned)((NUMBASE + NUMQUERY) * DIMENSION)];
	int* sparse_marker = new int[(NUMBASE + NUMQUERY) + 1];
	readSparse(BASEFILE, 0, (unsigned)(NUMBASE + NUMQUERY), sparse_indice, sparse_val, sparse_marker, (unsigned)((NUMBASE + NUMQUERY) * DIMENSION));
	end = Clock::now();
	etime_0 = (end - begin).count() / 1000000;
	std::cout << "Completed, used " << etime_0 << "ms. \n";

	std::cout << "Adding data to hashtable / Preprocessing / Indexing ...\n";
	int hash_chunk = NUMBASE / NUMHASHBATCH;
	begin = Clock::now();
	for (int b = 0; b < NUMHASHBATCH; b++) {
		myReservoir->add(hash_chunk, sparse_indice, sparse_val, sparse_marker + b * hash_chunk + NUMQUERY);
		if (b % BATCHPRINT == 0) {
			end = Clock::now();
			etime_0 = (end - begin).count() / 1000000;
			std::cout << "Batch " << b << "(datapoint " << (b * hash_chunk + NUMQUERY) << "), already taken " <<
				etime_0 << " ms." << std::endl;
			myReservoir->checkTableMemLoad();
		}
	}
#if defined OPENCL_KSELECT || defined GPU_TABLE || defined OPENCL_HASHING
	/* When using GPU, to ensure accurate timing, make sure GPU tasks all finish before ending the timer. */
	clFinish(myReservoir->command_queue_gpu);
#endif
	end = Clock::now();
	etime_0 = (end - begin).count() / 1000000;
	std::cout << "Completed, used " << etime_0 << "ms. \n";

	std::cout << "Querying...\n";
	unsigned int *queryOutputs = new unsigned int[NUMQUERY * TOPK]();
	begin = Clock::now();
	myReservoir->ann(NUMQUERY, sparse_indice, sparse_val, sparse_marker, queryOutputs, TOPK);
#if defined OPENCL_KSELECT || defined GPU_TABLE || defined OPENCL_HASHING
	clFinish(myReservoir->command_queue_gpu);
#endif
	end = Clock::now();
	etime_0 = (end - begin).count() / 1000000;
	std::cout << "Queried " << NUMQUERY << " datapoints, used " << etime_0 << "ms. \n";

	const int nCnt = 10;
	int nList[nCnt] = { 1, 10, 20, 30, 32, 40, 50, 64, 100, TOPK };
	const int gstdCnt = 8;
	float gstdVec[gstdCnt] = { 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.50 };
	const int tstdCnt = 10;
	int tstdVec[tstdCnt] = { 1, 10, 20, 30, 32, 40, 50, 64, 100, TOPK };
	
	/* Quality evaluations. */
	similarityMetric(sparse_indice, sparse_val, sparse_marker,
		sparse_indice, sparse_val, sparse_marker + NUMQUERY, queryOutputs, gtruth_dist,
		NUMQUERY, TOPK, AVAILABLE_TOPK, nList, nCnt);
	similarityOfData(gtruth_dist, NUMQUERY, TOPK, AVAILABLE_TOPK, nList, nCnt);
	evaluate(queryOutputs, NUMQUERY, TOPK, gtruth_indice, gtruth_dist, AVAILABLE_TOPK, gstdVec, gstdCnt, tstdVec, tstdCnt, nList, nCnt);				// The number of n interested. 

	delete[] sparse_indice;
	delete[] sparse_val;
	delete[] sparse_marker;
	delete[] gtruth_indice;
	delete[] gtruth_dist;
	delete[] queryOutputs;
}

void benchmark_dense() {
	float etime_0, etime_1, etime_2;
	auto begin = Clock::now();
	auto end = Clock::now();

	std::cout << "Initializing data structures and random numbers ..." << std::endl;
	begin = Clock::now();
	LSH *hashFamily = new LSH(1, RANGE_POW, NUMTABLES, DIMENSION, SAMFACTOR);
	LSHReservoirSampler *myReservoir = new LSHReservoirSampler(hashFamily, RANGE_POW, NUMTABLES, RESERVOIR_SIZE,
		DIMENSION, RANGE_ROW_U, NUMBASE, QUERYPROBES, HASHINGPROBES, OCCUPANCY); // Initialize hashtables and other datastructures. 
#if defined OPENCL_HASHING
	hashFamily->clLSH(myReservoir->platforms, myReservoir->devices_gpu + CL_DEVICE_ID, myReservoir->context_gpu,
		myReservoir->program_gpu, myReservoir->command_queue_gpu);
#endif
	end = Clock::now();
	etime_0 = (end - begin).count() / 1000000;
	std::cout << "Completed, used " << etime_0 << "ms. \n";

	myReservoir->showParams(); // Print out parameters. 

	std::cout << "Reading groundtruth ... " << std::endl;
	unsigned int* gtruth_indice = new unsigned int[NUMQUERY * AVAILABLE_TOPK];
	float* gtruth_dist = new float[NUMQUERY * AVAILABLE_TOPK];
	readGroundTruthInt(GTRUTHINDICE, NUMQUERY, AVAILABLE_TOPK, gtruth_indice);
	readGroundTruthFloat(GTRUTHDIST, NUMQUERY, AVAILABLE_TOPK, gtruth_dist);
	std::cout << "Completed. \n";

	std::cout << "Reading data ... " << std::endl;
	begin = Clock::now();
#if defined (SIFT1B) || defined(SIFT10M)
	float* dense_vectors = new float[(unsigned)((NUMBASE + NUMQUERY) * DIMENSION)];
	bvecs_read(QUERYFILE, 0, NUMQUERY, dense_vectors);
	bvecs_read(BASEFILE, 0, NUMBASE, dense_vectors + DIMENSION * NUMQUERY);
#else // SIFT 1M and Below
	float* dense_vectors = new float[(unsigned)((NUMBASE + NUMQUERY) * DIMENSION)];
	fvecs_read(QUERYFILE, 0, NUMQUERY, dense_vectors);
	fvecs_read(BASEFILE, 0, NUMBASE, dense_vectors + DIMENSION * NUMQUERY);
#endif
	end = Clock::now();
	etime_0 = (end - begin).count() / 1000000;
	std::cout << "Completed, used " << etime_0 << "ms. \n";

	std::cout << "Centering ... " << std::endl;
	begin = Clock::now();
	for (int i = 0; i < (NUMBASE + NUMQUERY); i++) {
		zCentering(dense_vectors + i * DIMENSION, DIMENSION);
	}
	end = Clock::now();
	etime_0 = (end - begin).count() / 1000000;
	std::cout << "Centering Used " << etime_0 << "ms. \n";

	std::cout << "Adding data to hashtable / Preprocessing / Indexing ...\n";
	int hash_chunk = NUMBASE / NUMHASHBATCH;
	begin = Clock::now();
	for (int b = 0; b < NUMHASHBATCH; b++) {
		myReservoir->add(hash_chunk, dense_vectors + (NUMQUERY + b * hash_chunk) * DIMENSION);
		if (b % BATCHPRINT == 0) {
			end = Clock::now();
			etime_0 = (end - begin).count() / 1000000;
			std::cout << "Batch " << b << "(datapoint " << (b * hash_chunk + NUMQUERY) << "), already taken " <<
				etime_0 << " ms." << std::endl;
			//myReservoir->checkTableMemLoad();
		}
	}
#if defined OPENCL_KSELECT || defined GPU_TABLE || defined OPENCL_HASHING
	/* When using GPU, to ensure accurate timing, make sure GPU tasks all finish before ending the timer. */
	clFinish(myReservoir->command_queue_gpu);
#endif
	end = Clock::now();
	etime_0 = (end - begin).count() / 1000000;
	std::cout << "Completed, used " << etime_0 << "ms. \n";

	std::cout << "Querying...\n";
	unsigned int *queryOutputs = new unsigned int[NUMQUERY * TOPK]();
	begin = Clock::now();
	myReservoir->ann(NUMQUERY, dense_vectors, queryOutputs, TOPK);
#if defined OPENCL_KSELECT || defined GPU_TABLE || defined OPENCL_HASHING
	clFinish(myReservoir->command_queue_gpu);
#endif
	end = Clock::now();
	etime_0 = (end - begin).count() / 1000000;
	std::cout << "Queried " << NUMQUERY << " datapoints, used " << etime_0 << "ms. \n";

	const int nCnt = 10;
	int nList[nCnt] = { 1, 10, 20, 30, 32, 40, 50, 64, 100, TOPK };
	const int gstdCnt = 8;
	float gstdVec[gstdCnt] = { 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.50 };
	const int tstdCnt = 10;
	int tstdVec[tstdCnt] = { 1, 10, 20, 30, 32, 40, 50, 64, 100, TOPK };

	/* Quality evaluations. */
	similarityMetric(dense_vectors, dense_vectors + NUMQUERY * DIMENSION,
		queryOutputs, gtruth_dist, DIMENSION, NUMQUERY, TOPK, AVAILABLE_TOPK, nList, nCnt);
	similarityOfData(gtruth_dist, NUMQUERY, TOPK, AVAILABLE_TOPK, nList, nCnt);
	evaluate(queryOutputs, NUMQUERY, TOPK, gtruth_indice, gtruth_dist, AVAILABLE_TOPK, gstdVec, gstdCnt, tstdVec, tstdCnt, nList, nCnt);				// The number of n interested. 

	delete[] dense_vectors;
	delete[] gtruth_indice;
	delete[] gtruth_dist;
	delete[] queryOutputs;
}

void benchmark_doph(int TEST_DOPH) {
	float etime_0, etime_1, etime_2;
	auto begin = Clock::now();
	auto end = Clock::now();

	size_t numVecs = NUMBASE + NUMQUERY;

	int* sparse_indice = new int[(unsigned)(numVecs * DIMENSION)];
	float* sparse_val = new float[(unsigned)(numVecs * DIMENSION)];
	int* sparse_marker = new int[numVecs + 1];
	readSparse(BASEFILE, 0, (unsigned)numVecs, sparse_indice, sparse_val, sparse_marker, (unsigned)(numVecs * DIMENSION));

	std::cout << "Intializing hash function. " << std::endl;
	begin = Clock::now();
	LSH *hashFamily = new LSH(2, K, TEST_DOPH, RANGE_POW);
	end = Clock::now();
	etime_0 = (end - begin).count() / 1000000;
	std::cout << "Used " << etime_0 << "ms. \n";
	std::cout << "Completed intializing hash function. " << std::endl;

	std::cout << "Generating signed random projection. " << std::endl;
	begin = Clock::now();
	int hash_chunk = numVecs / NUMHASHBATCH;
	unsigned int* allprobsHash = new unsigned int[hash_chunk * TEST_DOPH];
	unsigned int* allprobsIdx = new unsigned int[hash_chunk * TEST_DOPH];
	size_t startVec, endVec;
	for (int b = 0; b < NUMHASHBATCH; b++) {
		if (b % BATCHPRINT == 0) {
			end = Clock::now();
			etime_0 = (end - begin).count() / 1000000;
			std::cout << "Batch " << b << "(vec " << (b * hash_chunk + NUMQUERY) << "), already taken " <<
				etime_0 << " ms." << std::endl;
		}
		hashFamily->getHash(allprobsHash, allprobsIdx,
			sparse_indice, sparse_val,
			sparse_marker + hash_chunk * b, hash_chunk, 1);
	}
	end = Clock::now();
	std::cout << "Used " << etime_0 << "ms. \n";
	std::cout << "Completed generating random projections. " << std::endl;
}

void benchmark_smartrp(int SMART_RP) {
	float etime_0, etime_1, etime_2;
	auto begin = Clock::now();
	auto end = Clock::now();

	size_t numHashesToGen = SMART_RP;
	size_t numVecs = NUMBASE + NUMQUERY;

	int* sparse_indice = new int[(unsigned)(numVecs * DIMENSION)];
	float* sparse_val = new float[(unsigned)(numVecs * DIMENSION)];
	int* sparse_marker = new int[numVecs + 1];
	readSparse(BASEFILE, 0, (unsigned)numVecs, sparse_indice, sparse_val, sparse_marker, (unsigned)(numVecs * DIMENSION));

	/* Generate random numbers. */
	std::cout << "Generating random numbers for random projection. " << std::endl;
	begin = Clock::now();
	// randBits - random bits deciding to add or subtract, contain randbits for numTable * numHashPerFamily * samSize. 
	short *randBits = new short[numHashesToGen * FULL_DIMENSION];
	srand(time(0));
#pragma omp parallel for
	for (int tb = 0; tb < numHashesToGen; tb++) {
		for (int j = 0; j < FULL_DIMENSION; j++) {
			if (rand() % 3 == 0) {		// For 1/3 chance, generate -1 or 1
				if (rand() % 2 == 0) {
					randBits[tb * numHashesToGen + j] = 1;
				}
				else {
					randBits[tb * numHashesToGen + j] = -1;
				}
			}
			else { // For 2/3 chance, generate 0
				randBits[tb * numHashesToGen + j] = 0;
			}
		}
	}

	end = Clock::now();
	etime_0 = (end - begin).count() / 1000000;
	std::cout << "Used " << etime_0 << "ms. \n";
	std::cout << "Completed generating random numbers for random projection. " << std::endl;

	std::cout << "Generating signed random projection. " << std::endl;
	begin = Clock::now();
	int hash_chunk = numVecs / NUMHASHBATCH;
	unsigned int numOutputsToGen = numHashesToGen * numVecs;
	float *outputs = new float[numOutputsToGen];
	size_t startVec, endVec;
	for (int b = 0; b < NUMHASHBATCH; b++) {
		if (b % BATCHPRINT == 0) {
			end = Clock::now();
			etime_0 = (end - begin).count() / 1000000;
			std::cout << "Batch " << b << "(vec " << (b * hash_chunk + NUMQUERY) << "), already taken " <<
				etime_0 << " ms." << std::endl;
		}

		/* Hash Generation */
#pragma omp parallel private(startVec, endVec)
#pragma omp parallel for
		for (int i = 0; i < hash_chunk; i++) {
			startVec = sparse_marker[hash_chunk * b + i];
			endVec = sparse_marker[hash_chunk * b + i + 1];

			//smartrp_batch(SMART_RP, FULL_DIMENSION, 
			//	sparse_indice + startVec, 
			//	sparse_val + startVec, 
			//	endVec - startVec,
			//	randBits, outputs + SMART_RP * (b * hash_chunk + i));

			for (int h = 0; h < numHashesToGen; h++) {
				outputs[(hash_chunk * b + i) * numHashesToGen + h] =
					//std::cout << 
					smartrp(sparse_indice + startVec, sparse_val + startVec, endVec - startVec, randBits + h * FULL_DIMENSION);
			}
		}

	}
	end = Clock::now();
	std::cout << "Used " << etime_0 << "ms. \n";
	std::cout << "Completed generating random projections. " << std::endl;
}