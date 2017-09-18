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
#include "benchmarking.h"

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include "FrequentItems.h"

using namespace std;



int main(void) {
	
#if defined(SPARSE_DATASET)
	benchmark_sparse();
#elif defined(DENSE_DATASET)
	benchmark_dense();
#endif
	return 0;
}

