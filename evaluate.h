
#ifndef EVALUATE_H
#define EVALUATE_H

#include <stdio.h>
#include <stdlib.h>
#include <unordered_set>
#include <map>
#include <algorithm>

void similarityOfData(float *groundTruthDist, unsigned int numQueries, unsigned int topk,
	unsigned int availableTopk, int *nList, int nCnt);

void similarityMetric(float *queries, float *bases, unsigned int *queryOutputs, float *groundTruthDist,
	unsigned int dimension, unsigned int numQueries, unsigned int topk, unsigned int availableTopk, int *nList,
	int nCnt);

void similarityMetric(int *queries_indice, float *queries_val, int *queries_marker,
	int *bases_indice, float *bases_val, int *bases_marker, unsigned int *queryOutputs, float *groundTruthDist,
	unsigned int numQueries, unsigned int topk, unsigned int availableTopk, int *nList,
	int nCnt);

void evaluate(
	unsigned int *queryOutputs,		// The output indices of queries. 
	int numQueries,			// The number of query entries, should be the same for outputs and groundtruths. 
	int topk,				// The topk per query contained in the queryOutputs. 
	unsigned int *groundTruthIdx,	// The groundtruth indice vector. 
	float *groundTruthDist,	// The groundtruth distance vector. 
	int availableTopk,		// Available topk information in the groundtruth. 
	float *gstdVec,			// The goldstandards (similarity to be tested, a vector). 
	int gstdCnt,			// The number of goldstandards. 
	int *tstdVec,			// The Tstandards (top k gtruth to be tested, a vector). 
	int tstdCnt,			// The number of Tstandards. 
	int *nList,				// The n of R@n, T@n or G@n interested, a vector. 
	int nCnt);				// The number of n(s) interested. 

void gMetric(unsigned int *queryOutputs, int numQueries, int topk,
	unsigned int *groundTruthIdx, float *groundTruthDist, int availableTopk, float *gstdVec, const int gstdCnt, int *nList, int nCnt);

void rMetric(unsigned int *queryOutputs, int numQueries, int topk,
	unsigned int *groundTruthIdx, int availableTopk, int *nList, int nCnt);

void tMetric(unsigned int *queryOutputs, int numQueries, int topk,
	unsigned int *groundTruthIdx, float *groundTruthDist, int availableTopk, int *tstdVec, const int tstdCnt);

#endif /* EVALUATE_H */