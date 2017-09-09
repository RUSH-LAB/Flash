
#include "LSHReservoirSampler_config.h"
#include "indexing.h"

__kernel void reservoir_sampling_recur(
	__global uint *tableMem,
	__global uint *tablePointers,
	__global uint *tableMemAllocator,
	__global uint *allprobsHash,
	__global uint *allprobsIdx,
	__global uint *storelog,
	__global uint *reservoirRand,
	uint numReservoirsHashed,
	uint numProbePerTb,
	uint aggNumReservoirs,
	uint numRands,
	uint sechash_a,
	uint sechash_b, 
	uint _reservoirSize,
	uint _numSecHash) {

	uint probeIdx = get_global_id(0); 
	uint tb = get_global_id(1); 

	int TB = numProbePerTb * tb;

	uint hashIdx = allprobsHash[allprobsHashSimpleIdx(numProbePerTb, tb, probeIdx)];
	uint inputIdx = allprobsIdx[allprobsHashSimpleIdx(numProbePerTb, tb, probeIdx)];	
	uint ct = 0;
	
	/* Allocate the reservoir if non-existent. */ 
	/* Only lock the pointer when its not allocated. */
	uint allocIdx = atomic_cmpxchg(tablePointers + tablePointersIdx(numReservoirsHashed, hashIdx, tb, sechash_a, sechash_b), TABLENULL, 0);
	if (allocIdx != TABLENULL) {

		// ATOMIC: Obtain the counter, and increment the counter. (Counter initialized to 0 automatically). 
		// Counter counts from 0 to currentCount-1. 
		uint counter = atom_inc(tableMem + tableMemCtIdx(tb, allocIdx, aggNumReservoirs)); 

		// The counter here is the old counter. Current count is already counter + 1. 
		// If current count is larger than _reservoirSize, current item needs to be sampled. 
		uint reservoir_full = (counter + 1) > _reservoirSize;
	
		uint reservoirRandNum = reservoirRand[min(numRands,counter)]; // Overflow prevention. 

		uint location = reservoir_full * (reservoirRandNum) + (1 - reservoir_full) * counter;

		storelog[storelogIdIdx(numProbePerTb, probeIdx, tb)] = inputIdx;
		storelog[storelogCounterIdx(numProbePerTb, probeIdx, tb)] = counter;
		storelog[storelogLocationIdx(numProbePerTb, probeIdx, tb)] = location;
		storelog[storelogHashIdxIdx(numProbePerTb, probeIdx, tb)] = hashIdx; 

	} else { 
		allocIdx = atom_inc(tableMemAllocator + tableMemAllocatorIdx(tb)); 
		tablePointers[tablePointersIdx(numReservoirsHashed, hashIdx, tb, sechash_a, sechash_b)] = allocIdx;

		// ATOMIC: Obtain the counter, and increment the counter. (Counter initialized to 0 automatically). 
		// Counter counts from 0 to currentCount-1. 
		uint counter = atom_inc(tableMem + tableMemCtIdx(tb, allocIdx, aggNumReservoirs)); 

		// The counter here is the old counter. Current count is already counter + 1. 
		// If current count is larger than _reservoirSize, current item needs to be sampled. 
		uint reservoir_full = (counter + 1) > _reservoirSize;
	
		uint reservoirRandNum = reservoirRand[min(numRands,counter)]; // Overflow prevention. 

		uint location = reservoir_full * (reservoirRandNum) + (1 - reservoir_full) * counter;

		storelog[storelogIdIdx(numProbePerTb, probeIdx, tb)] = inputIdx;
		storelog[storelogCounterIdx(numProbePerTb, probeIdx, tb)] = counter;
		storelog[storelogLocationIdx(numProbePerTb, probeIdx, tb)] = location;
		storelog[storelogHashIdxIdx(numProbePerTb, probeIdx, tb)] = hashIdx; 
	}

}

/*
This kernel processes the storelog.
*/
__kernel void add_table(
	__global uint *tablePointers,
	__global uint *tableMem,
	__global uint *storelog,
	uint numProbePerTb,
	uint numReservoirsHashed,
	uint aggNumReservoirs,
	uint idBase,
	uint sechash_a,
	uint sechash_b,
	uint _reservoirSize, 
	uint _numSecHash) {

	uint tb = get_global_id(0);
	uint probeIdx = get_global_id(1);
	uint id = storelog[storelogIdIdx(numProbePerTb, probeIdx, tb)];
	uint hashIdx = storelog[storelogHashIdxIdx(numProbePerTb, probeIdx, tb)];
	uint allocIdx = tablePointers[max((unsigned)0,(unsigned)tablePointersIdx(numReservoirsHashed, hashIdx, tb, sechash_a, sechash_b))];
	/* TODO: access uncoalesced, workgroup size not optimized. */

	// If item_i spills out of the reservoir, it is capped to the dummy location at _reservoirSize. 
	uint locCapped = storelog[storelogLocationIdx(numProbePerTb, probeIdx, tb)];

	if (locCapped < _reservoirSize) {
		tableMem[tableMemResIdx(tb, allocIdx, aggNumReservoirs) + locCapped] = id + idBase;
	}
}

__kernel void extract_rows(
	__global uint *tablePointers, 
	__global uint *tableMem, 
	__global uint *hashIndices, 
	__global uint *queue,
	uint numReservoirsHashed, 
	uint aggNumReservoirs,
	uint numQueryEntries,
	uint segmentSizePow2, 
	uint sechash_a, 
	uint sechash_b,
	uint _reservoirSize, 
	uint _numSecHash, 
	uint _queryProbes) {

	uint queryIdx = get_global_id(0);
	uint tb = get_global_id(1);
	uint elemIdx = get_global_id(2);
	uint hashIdx;
	uint allocIdx;

	for (uint k = 0; k < _queryProbes; k++) {
		hashIdx = hashIndices[allprobsHashIdx(_queryProbes, numQueryEntries, tb, queryIdx, k)];
		allocIdx = tablePointers[tablePointersIdx(numReservoirsHashed, hashIdx, tb, sechash_a, sechash_b)];
		if (allocIdx != TABLENULL) {
			queue[queueElemIdx(segmentSizePow2, tb, queryIdx, k, elemIdx)] = 
				tableMem[tableMemResIdx(tb, allocIdx, aggNumReservoirs) + elemIdx];
		}
	}
}

#define isodd(number) (1 & (unsigned int)number)
__kernel void take_topk(
	__global uint *tally,
	__global uint *tallyCount, // Used to hold the new result. 
	uint segmentSizePow2, 
	uint topk) {
	
	unsigned int grpId = get_group_id(0); 
	uint localId = get_local_id(0); // ID in the topk. 

	uint myCopyLoc = isodd(grpId) ? 
		(topkIdx(segmentSizePow2, grpId) + localId) : 
		(topkIdx(segmentSizePow2, grpId) + localId + (segmentSizePow2 - topk)); 

	tallyCount[topkIdx(topk, grpId) + localId] = tally[myCopyLoc];
}

__kernel void mark_diff(
	__global uint *tally, 
	__global uint *tallyCount, // Where to store the location of change. 
	uint segmentSize, 
	uint _segmentSizeModulor) {
	
	uint gIdx = get_global_id(0);
	uint localQueueIdx = gIdx & _segmentSizeModulor; 
	
	/* Detect changes in the queue, and record where the change occurs. */
	if (localQueueIdx != 0) {
		tallyCount[gIdx] = (tally[gIdx] != tally[gIdx - 1]) ? gIdx : -1;
	}
	else { // The first element, no spot of comparison. 
		tallyCount[gIdx] = gIdx;
	}
}

/*
This kernel takes the difference-marked queue, and compact that of each query to the
front part of each segment. 

Global Size: allSegmentSize / l_segSize
Local Size: 1024 / l_segSize (assuming local memory size is 4096). 

l_segSize: The number of elements that each work item will go through. 
wg_segSize: an integral portion of a segment that is tallied by a workgroup. 

*/
__kernel void agg_diff(
	__global uint *tally,
	__global uint *tallyCount, // Where to store the location of change. 
	__global uint *g_queryCt,
	__local uint *localSegment, // Having size wg_segSize. 
	__local uint *localSegmentCompact, // Having size wg_segSize. 
	__local uint *localSegmentCnt, 
	__local uint *localSegmentCntCompact,
	__local uint *zeroSeg,	// Element counter for the query. 
	__local uint *localCt, 
	__local uint *queryCt,	// Element counter for the query. 
	uint segmentSizePow2) {

	uint wgSize = get_local_size(0); 
	uint localIdx = get_local_id(0); 
	uint queryIdx = get_group_id(0);
	uint i, l_offset, ct, cnt;
	event_t wait[4]; 
	queryCt[0] = 0; // Current wg compact length. 
	queryCt[1] = 0; // Cummulative query compact length. 

	/* zeroSeg, for initializing global mem to 0 vector. */
	l_offset = l_segSize * localIdx;
	for (i = 0; i < l_segSize; i++) { 
		zeroSeg[l_offset + i] = 0;
	}

	for (uint wgIdx = 0; wgIdx < (segmentSizePow2 / wg_segSize); wgIdx ++) { 
	
		// Copy workgroup segment to local memory. 
		wait[0] = async_work_group_copy(
			localSegment,
			tally + queryIdx * segmentSizePow2 +  wgIdx * wg_segSize,
			wg_segSize,
			0);
		wait[1] = async_work_group_copy(
			localSegmentCnt,
			tallyCount + queryIdx * segmentSizePow2 + wgIdx * wg_segSize,
			wg_segSize,
			0);

		wait_group_events(2, wait);

		/* Clear the global wg segment, which is already in the local memory. */
		wait[0] = async_work_group_copy(
			tallyCount + queryIdx * segmentSizePow2 +  wgIdx * wg_segSize,
			zeroSeg,
			wg_segSize,
			0);
		wait[1] = async_work_group_copy(
			tally + queryIdx * segmentSizePow2 +  wgIdx * wg_segSize,
			(__local uint *)zeroSeg,
			wg_segSize,
			0);
	
		ct = 0; // To count compact elements work item. 
		l_offset = localIdx * l_segSize; // Element offset in the workgroup segment. 
		for (i = 0; i < l_segSize; i++) {
			/* If tallyCount is not zero, something is there. Increment the counter and compact-store it. */
			cnt = localSegmentCnt[l_offset + i]; 
			/* TODO: Consider changing to "if", might be faster. Consider combining localseg and localcompact. */
			localSegmentCompact[l_offset + ct] = (cnt != -1) ? localSegment[l_offset + i] : 0;
			localSegmentCntCompact[l_offset + ct] = (cnt != -1) ? cnt : 0;
			ct += (cnt != -1) ? 1 : 0; 
		}

		/* ct is the size of each work element compact. */
		localCt[localIdx] = ct; // Record the lsegment compact length for each work item. 
		barrier(CLK_LOCAL_MEM_FENCE); 

		/* Each workitem finds out its own location in the the workgroup segment. */
		uint myLocalCompactOffset = 0;
		for (i = 0; i < localIdx; i++) {
			myLocalCompactOffset += localCt[i];
		}

		/* Each workitem copy the data in a compact format to the workgroup segment. */
		for (i = 0; i < ct; i++) {
			localSegment[myLocalCompactOffset + i] = localSegmentCompact[l_offset + i];
			localSegmentCnt[myLocalCompactOffset + i] = localSegmentCntCompact[l_offset + i];
		}

		/* Leading workitem finds out where the workgroup segment begin in the query segment. */
		if (localIdx == 0) { 
			queryCt[1] = 0; // Current length. 
			for (i = 0; i < wgSize; i++) {
				queryCt[1] += localCt[i];
			}
			queryCt[0] += queryCt[1]; // Where to end. 
		}

		wait_group_events(2, wait); /* Wait for clearing zero. */

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); 
		
		wait[2] = async_work_group_copy(
			tallyCount + queryIdx * segmentSizePow2 + queryCt[0] - queryCt[1],
			localSegmentCnt,
			queryCt[1],
			0);
		wait[3] = async_work_group_copy(
			tally + queryIdx * segmentSizePow2 + queryCt[0] - queryCt[1],
			localSegment,
			queryCt[1],
			0);

		wait_group_events(2, wait + 2); // Consider shifting out of for-loop. 
	}

	/* Update final offset of each query to the global counter .*/
	if (localIdx == 0) { 
		g_queryCt[queryIdx] = queryCt[0];
	}
}

/*
	This kernel subtracts the marked difference and calculates the counts. 
*/
__kernel void subtract_diff(
	__global uint *tally, 
	__global uint *tallyCount,
	__global uint *tallyBuffer,
	__global uint *g_queryCt,
	uint segmentSize, uint segmentSizePow2, uint _segmentSizeModulor, uint _segmentSizeBitShiftDivisor) {

	uint gIdx = get_global_id(0); 
	uint localQueueIdx = gIdx & _segmentSizeModulor; 
	uint queryIdx = gIdx >> _segmentSizeBitShiftDivisor; 

	if (localQueueIdx < (g_queryCt[queryIdx] - 1)) { // If is in the valid range. 
		tallyCount[gIdx] = tallyBuffer[gIdx + 1] - tallyBuffer[gIdx];
	}
	else if (localQueueIdx == (g_queryCt[queryIdx] - 1)) { // At the end of segment, finish off with segmentSize. 
		tallyCount[gIdx] = (queryIdx) * segmentSizePow2 + segmentSizePow2 - tallyBuffer[gIdx]; 
	}

	//if (localQueueIdx == 0) tallyCount[gIdx] = queryIdx; // For debugging purpose. 
}

/* The naive approach, only serve as comparison. */ 
__kernel void talley_count(
	__global uint *talley,
	__global int *talleyCount,
	__global uint *queueSorted,
	int segmentSize) {

	int queryIdx = get_global_id(0); // Index of incoming query. 

	int Q = queryIdx * segmentSize;

	int ok;
	int count = 1;
	uint obj = queueSorted[Q];
	int idx = 0;

	/* Go through the queue and tally. */
	for (int i = 1; i < segmentSize; i++) { 
	// WAAAAAAAAAAAAAAAAAY too many uncoalesced accesses to global memory.  
		ok = (obj != queueSorted[Q + i]);

		talley[Q + idx] = ok ? obj : talley[Q + idx];
		talleyCount[Q + idx] = ok ? count : talleyCount[Q + idx];

		obj = ok ? queueSorted[Q + i] : obj;
		count = (1 - ok) * count;
		idx += ok;
		count++;
	}
	talley[Q + idx] = obj;
	talleyCount[Q + idx] = count;
}