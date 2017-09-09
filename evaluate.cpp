#include "evaluate.h"
#include "misc.h"

#include <iostream>

using namespace std;

/*
* Function:  evaluate
* --------------------
* Evaluate the results of a dataset using various metrics, prints the result
* 
*  returns: nothing
*/
void evaluate (
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
	int nCnt) {				// The number of n(s) interested. 

	rMetric(queryOutputs, numQueries, topk, groundTruthIdx, availableTopk, nList, nCnt);
	tMetric(queryOutputs, numQueries, topk, groundTruthIdx, groundTruthDist, availableTopk, tstdVec, tstdCnt);
	gMetric(queryOutputs, numQueries, topk, groundTruthIdx, groundTruthDist, availableTopk, gstdVec, gstdCnt, nList, nCnt);
}

void similarityOfData(float *groundTruthDist, unsigned int numQueries, unsigned int topk, 
	unsigned int availableTopk, int *nList, int nCnt) {

	float *gtruth_avg = new float[nCnt]();
	float *out_avt = new float[nCnt]();

	for (int i = 0; i < numQueries; i++) {
		for (int j = 0; j < topk; j++) {
			for (int n = 0; n < nCnt; n++) {
				if (j < nList[n])
					gtruth_avg[n] += groundTruthDist[i * availableTopk + j];
			}
		}
	}
	for (unsigned int n = 0; n < nCnt; n++) {
		printf("S@%d = %1.3f \n", nList[n],
			gtruth_avg[n] / (numQueries * nList[n]));
	}
	for (unsigned int n = 0; n < nCnt; n++) printf("%d ", nList[n]);
	printf("\n");
	for (unsigned int n = 0; n < nCnt; n++) printf("%1.3f ", gtruth_avg[n] / (numQueries * nList[n]));
	printf("\n"); printf("\n");
}

/* Simple comparison of average similarity between gtruth and the outputs. */
void similarityMetric(int *queries_indice, float *queries_val, int *queries_marker, 
	int *bases_indice, float *bases_val, int *bases_marker, unsigned int *queryOutputs, float *groundTruthDist, 
	unsigned int numQueries, unsigned int topk, unsigned int availableTopk, int *nList,
	int nCnt) {

	float *gtruth_avg = new float[nCnt]();
	float *out_avt = new float[nCnt]();
	std::cout << "[similarityMetric] Averaging gtruth. " << std::endl;
	/* Ground truth average. */
	for (int i = 0; i < numQueries; i++) {
		for (int j = 0; j < topk; j++) {
			for (int n = 0; n < nCnt; n++) {
				if (j < nList[n])
					gtruth_avg[n] += groundTruthDist[i * availableTopk + j];
			}
		}
	}

	std::cout << "[similarityMetric] Averaging output. " << std::endl;
	/* Output average. */
	for (int i = 0; i < numQueries; i++) {
		int startA, endA;
		startA = queries_marker[i];
		endA = queries_marker[i + 1];
		
		for (int j = 0; j < topk; j++) {
			int startB, endB;
			startB = bases_marker[queryOutputs[i * topk + j]];
			endB = bases_marker[queryOutputs[i * topk + j] + 1];
			float dist = cosineDist(queries_indice + startA, queries_val + startA, endA - startA,
				bases_indice + startB, bases_val + startB, endB - startB);
			for (int n = 0; n < nCnt; n++) {
				if (j < nList[n]) out_avt[n] += dist;
			}
		}
	}

	/* Print results. */
	printf("\nS@k = s_out(s_true): In top k, average output similarity (average groundtruth similarity). \n");
	for (unsigned int n = 0; n < nCnt; n++) {
		printf("S@%d = %1.3f (%1.3f) \n", nList[n],
			out_avt[n] / (numQueries * nList[n]),
			gtruth_avg[n] / (numQueries * nList[n]));
	}
	for (unsigned int n = 0; n < nCnt; n++) printf("%d ", nList[n]);
	printf("\n");
	for (unsigned int n = 0; n < nCnt; n++) printf("%1.3f ", out_avt[n] / (numQueries * nList[n]));
	printf("\n");
	for (unsigned int n = 0; n < nCnt; n++) printf("%1.3f ", gtruth_avg[n] / (numQueries * nList[n]));
	printf("\n"); printf("\n");

}

/* Simple comparison of average similarity between gtruth and the outputs. */
void similarityMetric(float *queries, float *bases, unsigned int *queryOutputs, float *groundTruthDist,
	unsigned int dimension, unsigned int numQueries, unsigned int topk, unsigned int availableTopk, int *nList,	
	int nCnt) {

	float *gtruth_avg = new float[nCnt]();
	float *out_avt = new float[nCnt]();

	/* Ground truth average. */
	for (int i = 0; i < numQueries; i++) {
		for (int j = 0; j < topk; j++) {
			for (int n = 0; n < nCnt; n++) {
				if (j < nList[n])
					gtruth_avg[n] += groundTruthDist[i * availableTopk + j];
			}
		}
	}

	/* Output average. */
	for (int i = 0; i < numQueries; i++) {
		for (int j = 0; j < topk; j++) {

			float dist = cosineDist(queries + dimension * i,
				bases + dimension * queryOutputs[i * topk + j], dimension);
			for (int n = 0; n < nCnt; n++) {
				if (j < nList[n]) out_avt[n] += dist;
			}
		}
	}

	/* Print results. */
	printf("\nS@k = s_out(s_true): In top k, average output similarity (average groundtruth similarity). \n");
	for (unsigned int n = 0; n < nCnt; n++) {
		printf("S@%d = %1.3f (%1.3f) \n", nList[n],
			out_avt[n] / (numQueries * nList[n]), 
			gtruth_avg[n] / (numQueries * nList[n]));
	}
	for (unsigned int n = 0; n < nCnt; n++) printf("%d ", nList[n]);
	printf("\n");
	for (unsigned int n = 0; n < nCnt; n++) printf("%1.3f ", out_avt[n] / (numQueries * nList[n]));
	printf("\n");
	for (unsigned int n = 0; n < nCnt; n++) printf("%1.3f ", gtruth_avg[n] / (numQueries * nList[n]));
	printf("\n"); printf("\n");


}

void gMetric(unsigned int *queryOutputs, int numQueries, int topk,
	unsigned int *groundTruthIdx, float *groundTruthDist, int availableTopk, float *gstdVec, const int gstdCnt, int *nList, const int nCnt) {

#define gcIdx(g, k, nCnt) (g * nCnt + k)

	printf("\nGg@k: Average recall of sim>g neighbors in k first results. \n");

	int *validQueryContribution = new int[gstdCnt * nCnt]();
	int *validQueryCnt = new int[gstdCnt * nCnt]();
	float *goldRecall = new float[gstdCnt * nCnt]();

	float goldRecallFractionTotal;

	for (int g = 0; g < gstdCnt; g++) { // For each gold standard metric. 

		for (int k = 0; k < nCnt; k++) { // For each topk count. 

			goldRecallFractionTotal = 0; 

			for (int i = 0; i < numQueries; i++) {

				// For the current query, insert all gold standard groundtruths. 
				// 1. Distance < gold standards. 
				// 2. Within the current topk count being evaluated. 
				// In other words, number of goldstd points will be <= current topk evaluated. 
				// And hence recall can never be > 1, but can max at 1. 
				unordered_set<unsigned int> goldStdGTruths;
				for (int j = 0; j < availableTopk; j++) {  
					if (groundTruthDist[i * availableTopk + j] > gstdVec[g] && j < nList[k]) {
						goldStdGTruths.insert(groundTruthIdx[i * availableTopk + j]);
					}
				}

				// For the current gold-standard and topk evaluated, whether the current query 
				// actually has any contribution, if yes, how many contributions. 
				validQueryCnt[gcIdx(g, k, nCnt)] += (goldStdGTruths.size() > 0) ? 1 : 0; 
				validQueryContribution[gcIdx(g, k, nCnt)] += (int) goldStdGTruths.size(); 

				// If the current query has contribution. 
				if (goldStdGTruths.size() > 0) { 
					unordered_set<unsigned int> qout(queryOutputs + i * topk, queryOutputs + i * topk + nList[k]);

					// Compute intersection. 
					int  intersectCnt = 0;
					for (const auto& elem : goldStdGTruths) {
						if (qout.find(elem) != qout.end()) { // Elem is in intersection. 
							intersectCnt++; 
						}
					}

					goldRecallFractionTotal += (float)intersectCnt / (float)goldStdGTruths.size();

				}

			} // END i - each query.  

			// Average queries' recall fraction. 
			goldRecall[gcIdx(g, k, nCnt)] = (float)goldRecallFractionTotal / (float)validQueryCnt[gcIdx(g, k, nCnt)];

		} // END k. 
	} // END g.
	
	for (int g = 0; g < gstdCnt; g++) {
		for (int k = 0; k < nCnt; k++) {
			printf("G%1.2f@%d = %1.3f (%d queries %d contributions)\n", 
				gstdVec[g], 
				nList[k],
				goldRecall[gcIdx(g, k, nCnt)], 
				validQueryCnt[gcIdx(g, k, nCnt)],
				validQueryContribution[gcIdx(g, k, nCnt)]);
		}
		for (int k = 0; k < nCnt; k++) printf("%d ", nList[k]);
		printf("\n");
		for (int k = 0; k < nCnt; k++) printf("%1.3f ", goldRecall[gcIdx(g, k, nCnt)]);
		printf("\n"); printf("\n");
	}

	delete[] goldRecall;
	delete[] validQueryCnt;
	delete[] validQueryContribution;
}

// Accuracy measure R@k: fraction of query where the nearest neighbor is in the top k result. 
void rMetric(unsigned int *queryOutputs, int numQueries, int topk, 
	unsigned int *groundTruthIdx, int availableTopk, int *nList, int nCnt) {

	printf("\nR@k: Average fraction of query where the nearest neighbor is in the k first results. \n");
	
	/* There are nCnts different standards that needs to be tested. 
	   good_counts keep track of the counts of queries that have their top-1 found in nList[nCnt]. */
	int *good_counts = new int[nCnt](); 
	
	unsigned int top_nn;
	
	for (int i = 0; i < numQueries; i++) {
		
		top_nn = groundTruthIdx[i * availableTopk];

		for (int j = 0; j < topk; j++) { // Look for top-1 in top-k. 

			if (top_nn == queryOutputs[i * topk + j]) {  // When top-1 is found. 
				for (int myN = 0; myN < nCnt; myN++) {	 // For each standard.  
					if (j < nList[myN]) {	// If standard is satisfied. 
						good_counts[myN]++; // Count this query. 
					}
				}
				goto next_q; // Force goto next query to ensure testing integrity. . 
			}

		}
	next_q:
		continue;
	}

	for (int myN = 0; myN < nCnt; myN++) {
		printf("R@%d = %1.3f \n", nList[myN], (float)good_counts[myN] / numQueries);
	}
	for (int myN = 0; myN < nCnt; myN++) printf("%d ", nList[myN]);
	printf("\n");
	for (int myN = 0; myN < nCnt; myN++) printf("%1.3f ", (float)good_counts[myN] / numQueries);
	printf("\n"); printf("\n");
	delete[] good_counts;
}

void tMetric(unsigned int *queryOutputs, int numQueries, int topk,
	unsigned int *groundTruthIdx, float *groundTruthDist, int availableTopk, int *tstdVec, const int tstdCnt) {

	printf("\nT@k Average fraction of top k nearest neighbors returned in k first results. \n");
	
	float *sumOfFraction = new float[tstdCnt]();

	for (int g = 0; g < tstdCnt; g++) { // For each test. 

		for (int i = 0; i < numQueries; i++) {
			
			unordered_set<unsigned int> topTGtruths(groundTruthIdx + i * availableTopk, groundTruthIdx + i * availableTopk + tstdVec[g]);
			unordered_set<unsigned int> topTOutputs(queryOutputs + i * topk, queryOutputs + i * topk + tstdVec[g]);
			
			float tmp = 0;
			for (const auto& elem : topTGtruths) {
				if (topTOutputs.find(elem) != topTOutputs.end()) { // If elem is found in the intersection. 
					tmp++;
				}
			}
			sumOfFraction[g] += tmp / (float)tstdVec[g];
		}
	}

	for (int g = 0; g < tstdCnt; g++) {
		printf("T@%d = %1.3f\n", tstdVec[g], 
			(float)sumOfFraction[g] / (float)numQueries); 
	}
	for (int g = 0; g < tstdCnt; g++) printf("%d ", tstdVec[g]);
	printf("\n");
	for (int g = 0; g < tstdCnt; g++) printf("%1.3f ", (float)sumOfFraction[g] / (float)numQueries);
	printf("\n"); printf("\n");

	delete[] sumOfFraction;

}