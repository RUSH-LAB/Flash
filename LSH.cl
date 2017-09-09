
#define UNIVERSAL_HASH(x,M,a,b) ((unsigned) (a * x + b) >> (32 - M))
#define BINARY_HASH(x,a,b) ((unsigned) (a * x + b) >> 31)

#define hashesOutputIdx(rangePow, numInputs, dataIdx, tb, hashInFamIdx) (tb * (numInputs * rangePow) + dataIdx * rangePow + hashInFamIdx)
#define hashIndicesOutputIdx(numTables, numProbes, numInputs, dataIdx, probeIdx, tb) (numInputs * numProbes * tb + dataIdx * numProbes + probeIdx)

__kernel void dense_rand_proj(
	__global uint *hashes, 
	__global const float *X,
	__global const short *randBits, 
	__global const uint *indices, 
	uint numInputEntries, 
	uint samSize, 
	uint dimension, 
	uint rangePow,
	uint groupHashingSize,
	__local uint *indicesLocal,
	__local short *randBitsLocal, 
	__local uint *hashesLocal) {

	uint g_inputBatchIdx = get_global_id(1); 
	uint g_tb = get_global_id(2); 
	uint numTables = get_global_size(2); 
	uint l_hashIdx = get_local_id(0); 

	// Copy indice and randbits into local memory. 
	event_t waits[3];
	waits[0] = async_work_group_copy(indicesLocal,
		indices + g_tb * rangePow * samSize,
		samSize * rangePow,
		0);

	waits[1] = async_work_group_copy(randBitsLocal,
		randBits + g_tb * rangePow * samSize,
		samSize * rangePow,
		0);

	wait_group_events(2, waits);

	float value = 0;	// Value for each hash (1 bit in the global hashIdx). 
	uint ok = 0;			// Subtract / add binary flag. 
	uint indice = 0;		// The random index (for hashing) of a input vector, range [0, dimension - 1]. 
	float elem;
	uint inputIdx;

	for (uint i = 0; i < groupHashingSize; i++) {
		inputIdx = g_inputBatchIdx * groupHashingSize + i;
		value = 0;

		// Iterate 1/samFactor of dimension. 
		for (uint k = 0; k < samSize; k++) {
			indice = indicesLocal[l_hashIdx * samSize + k];
			ok = randBitsLocal[l_hashIdx * samSize + k] >= 0;
			elem = X[inputIdx * dimension + indice];
			value += ok ? elem : (-elem);
		}

		hashesLocal[i * rangePow + l_hashIdx] = value > 0;
	}
	
	event_t copyback = async_work_group_copy(
		hashes + g_tb * (numInputEntries * rangePow) + g_inputBatchIdx * groupHashingSize * rangePow,
		hashesLocal,
		groupHashingSize * rangePow,
		0);

	wait_group_events(1, &copyback);
}

__kernel void sparse_rand_proj(
	__global uint *hashes,
	__global const uint *dataIdx,
	__global const float *dataVal,
	__global const uint *dataMarker,
	__global const uint *hash_a,
	__global const uint *hash_b,
	__global const uint *binhash_a,
	__global const uint *binhash_b,
	uint numInputEntries, 
	uint rangePow,
	uint samFactor,
	uint groupHashingSize,
	__local uint *hashesLocal) {
	
	uint g_inputBatchIdx = get_global_id(1); 
	uint g_tb = get_global_id(2); 
	uint numTables = get_global_size(2); 
	uint l_hashIdx = get_local_id(0); 

	float value = 0; 	// Value for each hash (1 bit in the global hashIdx). 
	uint ok = 0;			// Subtract / add binary flag. 
	uint indice = 0;		// The random index (for hashing) of a input vector, range [0, dimension - 1]. 
	uint inputIdx; 
	uint sparseLen; 
	uint start = 0; 
	
	uint a1 = hash_a[g_tb * rangePow + l_hashIdx];
	uint b1 = hash_b[g_tb * rangePow + l_hashIdx];
	uint a2 = binhash_a[g_tb * rangePow + l_hashIdx];
	uint b2 = binhash_b[g_tb * rangePow + l_hashIdx];

	// Process groupHashingSize number of inputs in one workgroup call.
	for (uint i = 0; i < groupHashingSize; i++) {
		inputIdx = g_inputBatchIdx * groupHashingSize + i;
		value = 0;
		
		start = dataMarker[inputIdx];
		sparseLen = dataMarker[inputIdx+1] - start;
		
		// Iterate 1/samFactor of dimension. 
		for (uint k = 0; k < sparseLen; k++) {
			indice = dataIdx[start + k]; // This step is TOO slow, especially when sparseLen is large.
			ok = BINARY_HASH(indice,a2,b2);
			value += (UNIVERSAL_HASH(indice,samFactor,a1,b1) == 1) ? (ok ? dataVal[start + indice] : (-dataVal[start + indice])) : 0;
		}
		hashesLocal[i * rangePow + l_hashIdx] = value > 0;
	}
	
	event_t copyback = async_work_group_copy(
		hashes + g_tb * (numInputEntries * rangePow) + g_inputBatchIdx * groupHashingSize * rangePow,
		hashesLocal,
		groupHashingSize * rangePow,
		0);

	wait_group_events(1, &copyback);
}

__kernel void mult_probes_storeid(
	__global uint *allprobsHash,
	__global uint *allprobsIdx,
	__global uint *hashes,
	uint numInputEntries,
	uint rangePow,
	uint numTables,
	uint numProbes) {

	uint inputIdx = get_global_id(0);
	uint tb = get_global_id(1);

	uint hashIdx = 0;
	for (uint k = 0; k < rangePow; k++) {
		// First hashbit is the smallest bit. 
		hashIdx |= (unsigned) hashes[hashesOutputIdx(rangePow, numInputEntries, inputIdx, tb, k)] << k;
	}
	allprobsHash[hashIndicesOutputIdx(numTables, numProbes, numInputEntries, inputIdx, 0, tb)] = hashIdx;  
	allprobsIdx[hashIndicesOutputIdx(numTables, numProbes, numInputEntries, inputIdx, 0, tb)] = inputIdx;
	for (uint k = 1; k < numProbes; k++) {
		allprobsHash[hashIndicesOutputIdx(numTables, numProbes, numInputEntries, inputIdx, k, tb)] = hashIdx ^ (1 << (k-1));  
		allprobsIdx[hashIndicesOutputIdx(numTables, numProbes, numInputEntries, inputIdx, k, tb)] = inputIdx;
	}
}

__kernel void mult_probes(
	__global uint *allprobsHash,
	__global uint *hashes,
	uint numInputEntries,
	uint rangePow,
	uint numTables,
	uint numProbes) {

	uint inputIdx = get_global_id(0);
	uint tb = get_global_id(1);

	uint hashIdx = 0;
	for (uint k = 0; k < rangePow; k++) {
		// First hashbit is the smallest bit. 
		hashIdx |= (unsigned) hashes[hashesOutputIdx(rangePow, numInputEntries, inputIdx, tb, k)] << k;
	}
	allprobsHash[hashIndicesOutputIdx(numTables, numProbes, numInputEntries, inputIdx, 0, tb)] = hashIdx;  
	for (uint k = 1; k < numProbes; k++) {
		allprobsHash[hashIndicesOutputIdx(numTables, numProbes, numInputEntries, inputIdx, k, tb)] = hashIdx ^ (1 << (k-1));  
	}
}