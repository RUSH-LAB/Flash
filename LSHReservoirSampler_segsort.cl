
/* Code adapted from MANNING - OpenCL in action tutorial - bitonic sort. 
	These kernels perform segmented sort. 
*/

/* Sort elements within a vector */
#define VECTOR_SORT(input, dir)                                   \
   comp = input < shuffle(input, mask2) ^ dir;                    \
   input = shuffle(input, as_uint4(comp * 2 + add2));             \
   comp = input < shuffle(input, mask1) ^ dir;                    \
   input = shuffle(input, as_uint4(comp + add1));                 \

#define VECTOR_SORT_KV(input, input_v, dir)                       \
   comp = input < shuffle(input, mask2) ^ dir;                    \
   input = shuffle(input, as_uint4(comp * 2 + add2));             \
   input_v = shuffle(input_v, as_uint4(comp * 2 + add2));         \
   comp = input < shuffle(input, mask1) ^ dir;                    \
   input = shuffle(input, as_uint4(comp + add1));                 \
   input_v = shuffle(input_v, as_uint4(comp + add1));             \

#define VECTOR_SWAP(input1, input2, dir)                          \
   temp = input1;                                                 \
   comp = (input1 < input2 ^ dir) * 4 + add3;                     \
   input1 = shuffle2(input1, input2, as_uint4(comp));             \
   input2 = shuffle2(input2, temp, as_uint4(comp));               \

#define VECTOR_SWAP_KV(input1, input2, input1_v, input2_v, dir)   \
   comp = (input1 < input2 ^ dir) * 4 + add3;                     \
   temp = input1;                                                 \
   input1 = shuffle2(input1, input2, as_uint4(comp));             \
   input2 = shuffle2(input2, temp, as_uint4(comp));               \
   temp_v = input1_v;											  \
   input1_v = shuffle2(input1_v, input2_v, as_uint4(comp));       \
   input2_v = shuffle2(input2_v, temp_v, as_uint4(comp));         \


/* Perform initial sort */
__kernel void bsort_preprocess_kv(__global uint4 *g_data, __global uint4 *g_data_v, uint valMax) {

	uint id, global_start;
	uint4 input1, input2, input1_v, input2_v;

	id = get_local_id(0) * 2; // Each work-item sorts two 4-vectors, total of 8 elements. 
	global_start = get_group_id(0) * get_local_size(0) * 2 + id;

	// Copy two vector-4 from the global memory. 
	input1 = g_data[global_start];
	input2 = g_data[global_start + 1];
	input1_v = g_data_v[global_start];
	input2_v = g_data_v[global_start + 1];

	input1.x = input1.x * valMax + input1_v.x;
	input1.y = input1.y * valMax + input1_v.y;
	input1.z = input1.z * valMax + input1_v.z;
	input1.w = input1.w * valMax + input1_v.w;
	input2.x = input2.x * valMax + input2_v.x;
	input2.y = input2.y * valMax + input2_v.y;
	input2.z = input2.z * valMax + input2_v.z;
	input2.w = input2.w * valMax + input2_v.w;

	g_data[global_start] = input1;
	g_data[global_start + 1] = input2;
	g_data_v[global_start] = input1_v;
	g_data_v[global_start + 1] = input2_v;
}

__kernel void bsort_postprocess_kv(__global uint4 *g_data, __global uint4 *g_data_v, uint valMax) {

	uint id, global_start;
	uint4 input1, input2, input1_v, input2_v;

	id = get_local_id(0) * 2; // Each work-item sorts two 4-vectors, total of 8 elements. 
	global_start = get_group_id(0) * get_local_size(0) * 2 + id;

	// Copy two vector-4 from the global memory. 
	input1 = g_data[global_start];
	input2 = g_data[global_start + 1];
	input1_v = g_data_v[global_start];
	input2_v = g_data_v[global_start + 1];

	input1.x = (input1.x - input1_v.x) / valMax;
	input1.y = (input1.y - input1_v.y) / valMax;
	input1.z = (input1.z - input1_v.z) / valMax;
	input1.w = (input1.w - input1_v.w) / valMax;
	input2.x = (input2.x - input2_v.x) / valMax;
	input2.y = (input2.y - input2_v.y) / valMax;
	input2.z = (input2.z - input2_v.z) / valMax;
	input2.w = (input2.w - input2_v.w) / valMax;

	g_data[global_start] = input1;
	g_data[global_start + 1] = input2;
	g_data_v[global_start] = input1_v;
	g_data_v[global_start + 1] = input2_v;
}

/* Perform initial sort */
__kernel void bsort_init_manning_kv(__global uint4 *g_data, __local uint4 *l_data, 
	__global uint4 *g_data_v, __local uint4 *l_data_v) {

	int dir;
	uint id, global_start, size, stride;
	uint4 input1, input2, temp;
	uint4 input1_v, input2_v, temp_v;
	int4 comp;

	uint4 mask1 = (uint4)(1, 0, 3, 2);
	uint4 mask2 = (uint4)(2, 3, 0, 1);
	uint4 mask3 = (uint4)(3, 2, 1, 0);

	int4 add1 = (int4)(1, 1, 3, 3);
	int4 add2 = (int4)(2, 3, 2, 3);
	int4 add3 = (int4)(1, 2, 2, 3);

	id = get_local_id(0) * 2; // Each work-item sorts two 4-vectors, total of 8 elements. 
	global_start = get_group_id(0) * get_local_size(0) * 2 + id;

	// Copy two vector-4 from the global memory. 
	input1 = g_data[global_start];
	input2 = g_data[global_start + 1];
	input1_v = g_data_v[global_start];
	input2_v = g_data_v[global_start + 1];
	// printf("%u %u %u %u ", input1_v.x, input1_v.y, input1_v.z, input1_v.w);

	/* Sort input 1 - ascending */ /* For, values, only shuffle as is, do not compare. */
	comp = input1 < shuffle(input1, mask1);
	input1 = shuffle(input1, as_uint4(comp + add1));
	input1_v = shuffle(input1_v, as_uint4(comp + add1));
	comp = input1 < shuffle(input1, mask2);
	input1 = shuffle(input1, as_uint4(comp * 2 + add2));
	input1_v = shuffle(input1_v, as_uint4(comp * 2 + add2));
	comp = input1 < shuffle(input1, mask3);
	input1 = shuffle(input1, as_uint4(comp + add3));
	input1_v = shuffle(input1_v, as_uint4(comp + add3));

	/* Sort input 2 - descending */
	comp = input2 > shuffle(input2, mask1);
	input2 = shuffle(input2, as_uint4(comp + add1));
	input2_v = shuffle(input2_v, as_uint4(comp + add1));
	comp = input2 > shuffle(input2, mask2);
	input2 = shuffle(input2, as_uint4(comp * 2 + add2));
	input2_v = shuffle(input2_v, as_uint4(comp * 2 + add2));
	comp = input2 > shuffle(input2, mask3);
	input2 = shuffle(input2, as_uint4(comp + add3));
	input2_v = shuffle(input2_v, as_uint4(comp + add3));

	/* Swap corresponding elements of input 1 and 2 */
	add3 = (int4)(4, 5, 6, 7);
	dir = get_local_id(0) % 2 * -1;
	temp = input1;
	comp = (input1 < input2 ^ dir) * 4 + add3;
	input1 = shuffle2(input1, input2, as_uint4(comp));
	input2 = shuffle2(input2, temp, as_uint4(comp));
	temp_v = input1_v;
	input1_v = shuffle2(input1_v, input2_v, as_uint4(comp));
	input2_v = shuffle2(input2_v, temp_v, as_uint4(comp));

	/* Sort data and store in local memory */
	VECTOR_SORT_KV(input1, input1_v, dir);
	VECTOR_SORT_KV(input2, input2_v, dir);
	l_data[id] = input1;
	l_data[id + 1] = input2;
	l_data_v[id] = input1_v;
	l_data_v[id + 1] = input2_v;

	/* Create bitonic set */
	
	// Outer stages. 
	for (size = 2; size < get_local_size(0); size <<= 1) {
		dir = (get_local_id(0) / size & 1) * -1;

		// Inner stages. 
		for (stride = size; stride > 1; stride >>= 1) {
			barrier(CLK_LOCAL_MEM_FENCE);
			id = get_local_id(0) + (get_local_id(0) / stride)*stride;
			VECTOR_SWAP_KV(l_data[id], l_data[id + stride], l_data_v[id], l_data_v[id + stride], dir)
		}

		barrier(CLK_LOCAL_MEM_FENCE);
		id = get_local_id(0) * 2;
		input1 = l_data[id]; input2 = l_data[id + 1];
		input1_v = l_data_v[id]; input2_v = l_data_v[id + 1];
		comp = (input1 < input2 ^ dir) * 4 + add3;
		temp = input1;
		input1 = shuffle2(input1, input2, as_uint4(comp));
		input2 = shuffle2(input2, temp, as_uint4(comp));
		temp_v = input1_v;
		input1_v = shuffle2(input1_v, input2_v, as_uint4(comp));
		input2_v = shuffle2(input2_v, temp_v, as_uint4(comp));
		VECTOR_SORT_KV(input1, input1_v, dir);
		VECTOR_SORT_KV(input2, input2_v, dir);
		l_data[id] = input1;
		l_data[id + 1] = input2;
		l_data_v[id] = input1_v;
		l_data_v[id + 1] = input2_v;
	}

	/* Perform bitonic merge */
	dir = (get_group_id(0) % 2) * -1;
	for (stride = get_local_size(0); stride > 1; stride >>= 1) {
		barrier(CLK_LOCAL_MEM_FENCE);
		id = get_local_id(0) + (get_local_id(0) / stride)*stride;
		VECTOR_SWAP_KV(l_data[id], l_data[id + stride], l_data_v[id], l_data_v[id + stride], dir)
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	/* Perform final sort */
	id = get_local_id(0) * 2;
	input1 = l_data[id]; input2 = l_data[id + 1];
	input1_v = l_data_v[id]; input2_v = l_data_v[id + 1];
	comp = (input1 < input2 ^ dir) * 4 + add3;
	temp = input1;
	input1 = shuffle2(input1, input2, as_uint4(comp));
	input2 = shuffle2(input2, temp, as_uint4(comp));
	temp_v = input1_v;
	input1_v = shuffle2(input1_v, input2_v, as_uint4(comp));
	input2_v = shuffle2(input2_v, temp_v, as_uint4(comp));
	VECTOR_SORT_KV(input1, input1_v, dir);
	VECTOR_SORT_KV(input2, input2_v, dir);

	g_data[global_start] = input1;
	g_data[global_start + 1] = input2;
	g_data_v[global_start] = input1_v;
	g_data_v[global_start + 1] = input2_v;
}

/* Perform initial sort */
__kernel void bsort_init_manning(__global uint4 *g_data, __local uint4 *l_data) {

	int dir;
	uint id, global_start, size, stride;
	uint4 input1, input2, temp;
	int4 comp;

	uint4 mask1 = (uint4)(1, 0, 3, 2);
	uint4 mask2 = (uint4)(2, 3, 0, 1);
	uint4 mask3 = (uint4)(3, 2, 1, 0);

	int4 add1 = (int4)(1, 1, 3, 3);
	int4 add2 = (int4)(2, 3, 2, 3);
	int4 add3 = (int4)(1, 2, 2, 3);

	id = get_local_id(0) * 2; // Each work-item sorts two 4-vectors, total of 8 elements. 
	global_start = get_group_id(0) * get_local_size(0) * 2 + id;

	// Copy two vector-4 from the global memory. 
	input1 = g_data[global_start];
	input2 = g_data[global_start + 1];

	/* Sort input 1 - ascending */
	comp = input1 < shuffle(input1, mask1);
	input1 = shuffle(input1, as_uint4(comp + add1));
	comp = input1 < shuffle(input1, mask2);
	input1 = shuffle(input1, as_uint4(comp * 2 + add2));
	comp = input1 < shuffle(input1, mask3);
	input1 = shuffle(input1, as_uint4(comp + add3));

	/* Sort input 2 - descending */
	comp = input2 > shuffle(input2, mask1);
	input2 = shuffle(input2, as_uint4(comp + add1));
	comp = input2 > shuffle(input2, mask2);
	input2 = shuffle(input2, as_uint4(comp * 2 + add2));
	comp = input2 > shuffle(input2, mask3);
	input2 = shuffle(input2, as_uint4(comp + add3));

	/* Swap corresponding elements of input 1 and 2 */
	add3 = (int4)(4, 5, 6, 7);
	dir = get_local_id(0) % 2 * -1;
	temp = input1;
	comp = (input1 < input2 ^ dir) * 4 + add3;
	input1 = shuffle2(input1, input2, as_uint4(comp));
	input2 = shuffle2(input2, temp, as_uint4(comp));

	/* Sort data and store in local memory */
	VECTOR_SORT(input1, dir);
	VECTOR_SORT(input2, dir);
	l_data[id] = input1;
	l_data[id + 1] = input2;

	/* Create bitonic set */

	// Outer stages. 
	for (size = 2; size < get_local_size(0); size <<= 1) {
		dir = (get_local_id(0) / size & 1) * -1;

		// Inner stages. 
		for (stride = size; stride > 1; stride >>= 1) {
			barrier(CLK_LOCAL_MEM_FENCE);
			id = get_local_id(0) + (get_local_id(0) / stride)*stride;
			VECTOR_SWAP(l_data[id], l_data[id + stride], dir)
		}

		barrier(CLK_LOCAL_MEM_FENCE);
		id = get_local_id(0) * 2;
		input1 = l_data[id]; input2 = l_data[id + 1];
		temp = input1;
		comp = (input1 < input2 ^ dir) * 4 + add3;
		input1 = shuffle2(input1, input2, as_uint4(comp));
		input2 = shuffle2(input2, temp, as_uint4(comp));
		VECTOR_SORT(input1, dir);
		VECTOR_SORT(input2, dir);
		l_data[id] = input1;
		l_data[id + 1] = input2;
	}

	/* Perform bitonic merge */
	dir = (get_group_id(0) % 2) * -1;
	for (stride = get_local_size(0); stride > 1; stride >>= 1) {
		barrier(CLK_LOCAL_MEM_FENCE);
		id = get_local_id(0) + (get_local_id(0) / stride)*stride;
		VECTOR_SWAP(l_data[id], l_data[id + stride], dir)
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	/* Perform final sort */
	id = get_local_id(0) * 2;
	input1 = l_data[id]; input2 = l_data[id + 1];
	temp = input1;
	comp = (input1 < input2 ^ dir) * 4 + add3;
	input1 = shuffle2(input1, input2, as_uint4(comp));
	input2 = shuffle2(input2, temp, as_uint4(comp));
	VECTOR_SORT(input1, dir);
	VECTOR_SORT(input2, dir);
	g_data[global_start] = input1;
	g_data[global_start + 1] = input2;
}

__kernel void bsort_stage_0_manning_kv(__global uint4 *g_data, __local uint4 *l_data,
	__global uint4 *g_data_v, __local uint4 *l_data_v,
	uint high_stage) {

	int dir;
	uint id, global_start, stride;
	uint4 input1, input2, temp, input1_v, input2_v, temp_v;
	int4 comp;

	uint4 mask1 = (uint4)(1, 0, 3, 2);
	uint4 mask2 = (uint4)(2, 3, 0, 1);
	uint4 mask3 = (uint4)(3, 2, 1, 0);

	int4 add1 = (int4)(1, 1, 3, 3);
	int4 add2 = (int4)(2, 3, 2, 3);
	int4 add3 = (int4)(4, 5, 6, 7);

	/* Determine data location in global memory */
	id = get_local_id(0);
	dir = (get_group_id(0) / high_stage & 1) * -1;
	global_start = get_group_id(0) * get_local_size(0) * 2 + id;

	/* Perform initial swap */
	input1 = g_data[global_start];
	input2 = g_data[global_start + get_local_size(0)];
	input1_v = g_data_v[global_start];
	input2_v = g_data_v[global_start + get_local_size(0)];
	comp = (input1 < input2 ^ dir) * 4 + add3;
	l_data[id] = shuffle2(input1, input2, as_uint4(comp));
	l_data[id + get_local_size(0)] = shuffle2(input2, input1, as_uint4(comp));
	l_data_v[id] = shuffle2(input1_v, input2_v, as_uint4(comp));
	l_data_v[id + get_local_size(0)] = shuffle2(input2_v, input1_v, as_uint4(comp));

	/* Perform bitonic merge */
	for (stride = get_local_size(0) / 2; stride > 1; stride >>= 1) {
		barrier(CLK_LOCAL_MEM_FENCE);
		id = get_local_id(0) + (get_local_id(0) / stride)*stride;
		VECTOR_SWAP_KV(l_data[id], l_data[id + stride], 
			l_data_v[id], l_data_v[id + stride], dir)
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	/* Perform final sort */
	id = get_local_id(0) * 2;
	input1 = l_data[id]; input2 = l_data[id + 1];
	input1_v = l_data_v[id]; input2_v = l_data_v[id + 1];
	temp = input1;
	comp = (input1 < input2 ^ dir) * 4 + add3;
	input1 = shuffle2(input1, input2, as_uint4(comp));
	input2 = shuffle2(input2, temp, as_uint4(comp));
	temp_v = input1_v;
	input1_v = shuffle2(input1_v, input2_v, as_uint4(comp));
	input2_v = shuffle2(input2_v, temp_v, as_uint4(comp));
	VECTOR_SORT_KV(input1, input1_v, dir);
	VECTOR_SORT_KV(input2, input2_v, dir);

	/* Store output in global memory */
	g_data[global_start + get_local_id(0)] = input1;
	g_data[global_start + get_local_id(0) + 1] = input2;
	g_data_v[global_start + get_local_id(0)] = input1_v;
	g_data_v[global_start + get_local_id(0) + 1] = input2_v;
}

/* Perform lowest stage of the bitonic sort */
__kernel void bsort_stage_0_manning(__global uint4 *g_data, __local uint4 *l_data,
	uint high_stage) {

	int dir;
	uint id, global_start, stride;
	uint4 input1, input2, temp;
	int4 comp;

	uint4 mask1 = (uint4)(1, 0, 3, 2);
	uint4 mask2 = (uint4)(2, 3, 0, 1);
	uint4 mask3 = (uint4)(3, 2, 1, 0);

	int4 add1 = (int4)(1, 1, 3, 3);
	int4 add2 = (int4)(2, 3, 2, 3);
	int4 add3 = (int4)(4, 5, 6, 7);

	/* Determine data location in global memory */
	id = get_local_id(0);
	dir = (get_group_id(0) / high_stage & 1) * -1;
	global_start = get_group_id(0) * get_local_size(0) * 2 + id;

	/* Perform initial swap */
	input1 = g_data[global_start];
	input2 = g_data[global_start + get_local_size(0)];
	comp = (input1 < input2 ^ dir) * 4 + add3;
	l_data[id] = shuffle2(input1, input2, as_uint4(comp));
	l_data[id + get_local_size(0)] = shuffle2(input2, input1, as_uint4(comp));

	/* Perform bitonic merge */
	for (stride = get_local_size(0) / 2; stride > 1; stride >>= 1) {
		barrier(CLK_LOCAL_MEM_FENCE);
		id = get_local_id(0) + (get_local_id(0) / stride)*stride;
		VECTOR_SWAP(l_data[id], l_data[id + stride], dir)
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	/* Perform final sort */
	id = get_local_id(0) * 2;
	input1 = l_data[id]; input2 = l_data[id + 1];
	temp = input1;
	comp = (input1 < input2 ^ dir) * 4 + add3;
	input1 = shuffle2(input1, input2, as_uint4(comp));
	input2 = shuffle2(input2, temp, as_uint4(comp));
	VECTOR_SORT(input1, dir);
	VECTOR_SORT(input2, dir);

	/* Store output in global memory */
	g_data[global_start + get_local_id(0)] = input1;
	g_data[global_start + get_local_id(0) + 1] = input2;
}

/* Perform successive stages of the bitonic sort */
__kernel void bsort_stage_n_manning_kv(__global uint4 *g_data, __local uint4 *l_data, 
	__global uint4 *g_data_v, __local uint4 *l_data_v,
	uint stage, uint high_stage) {

	int dir;
	uint4 input1, input2, input1_v, input2_v;
	int4 comp, add;
	uint global_start, global_offset;

	add = (int4)(4, 5, 6, 7);

	/* Determine location of data in global memory */
	dir = (get_group_id(0) / high_stage & 1) * -1;
	global_start = (get_group_id(0) + (get_group_id(0) / stage)*stage) *
		get_local_size(0) + get_local_id(0);
	global_offset = stage * get_local_size(0);

	/* Perform swap */
	input1 = g_data[global_start];
	input2 = g_data[global_start + global_offset];
	input1_v = g_data_v[global_start];
	input2_v = g_data_v[global_start + global_offset];
	comp = (input1 < input2 ^ dir) * 4 + add;
	g_data[global_start] = shuffle2(input1, input2, as_uint4(comp));
	g_data_v[global_start] = shuffle2(input1_v, input2_v, as_uint4(comp));
	g_data[global_start + global_offset] = shuffle2(input2, input1, as_uint4(comp));
	g_data_v[global_start + global_offset] = shuffle2(input2_v, input1_v, as_uint4(comp));
}

/* Perform successive stages of the bitonic sort */
__kernel void bsort_stage_n_manning(__global uint4 *g_data, __local uint4 *l_data,
	uint stage, uint high_stage) {

	int dir;
	uint4 input1, input2;
	int4 comp, add;
	uint global_start, global_offset;

	add = (int4)(4, 5, 6, 7);

	/* Determine location of data in global memory */
	dir = (get_group_id(0) / high_stage & 1) * -1;
	global_start = (get_group_id(0) + (get_group_id(0) / stage)*stage) *
		get_local_size(0) + get_local_id(0);
	global_offset = stage * get_local_size(0);

	/* Perform swap */
	input1 = g_data[global_start];
	input2 = g_data[global_start + global_offset];
	comp = (input1 < input2 ^ dir) * 4 + add;
	g_data[global_start] = shuffle2(input1, input2, as_uint4(comp));
	g_data[global_start + global_offset] = shuffle2(input2, input1, as_uint4(comp));
}

/* Sort the bitonic set */
__kernel void bsort_merge_manning(__global uint4 *g_data, __local uint4 *l_data, uint stage, int dir) {

	uint4 input1, input2;
	int4 comp, add;
	uint global_start, global_offset;

	add = (int4)(4, 5, 6, 7);

	/* Determine location of data in global memory */
	global_start = (get_group_id(0) + (get_group_id(0) / stage)*stage) *
		get_local_size(0) + get_local_id(0);
	global_offset = stage * get_local_size(0);

	/* Perform swap */
	input1 = g_data[global_start];
	input2 = g_data[global_start + global_offset];
	comp = (input1 < input2 ^ dir) * 4 + add;
	g_data[global_start] = shuffle2(input1, input2, as_uint4(comp));
	g_data[global_start + global_offset] = shuffle2(input2, input1, as_uint4(comp));
}

/* Perform final step of the bitonic merge */
__kernel void bsort_merge_last_manning(__global uint4 *g_data, __local uint4 *l_data, int dir) {

	uint id, global_start, stride;
	uint4 input1, input2, temp;
	int4 comp;

	uint4 mask1 = (uint4)(1, 0, 3, 2);
	uint4 mask2 = (uint4)(2, 3, 0, 1);
	uint4 mask3 = (uint4)(3, 2, 1, 0);

	int4 add1 = (int4)(1, 1, 3, 3);
	int4 add2 = (int4)(2, 3, 2, 3);
	int4 add3 = (int4)(4, 5, 6, 7);

	/* Determine location of data in global memory */
	id = get_local_id(0);
	global_start = get_group_id(0) * get_local_size(0) * 2 + id;

	/* Perform initial swap */
	input1 = g_data[global_start];
	input2 = g_data[global_start + get_local_size(0)];
	comp = (input1 < input2 ^ dir) * 4 + add3;
	l_data[id] = shuffle2(input1, input2, as_uint4(comp));
	l_data[id + get_local_size(0)] = shuffle2(input2, input1, as_uint4(comp));

	/* Perform bitonic merge */
	for (stride = get_local_size(0) / 2; stride > 1; stride >>= 1) {
		barrier(CLK_LOCAL_MEM_FENCE);
		id = get_local_id(0) + (get_local_id(0) / stride)*stride;
		VECTOR_SWAP(l_data[id], l_data[id + stride], dir)
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	/* Perform final sort */
	id = get_local_id(0) * 2;
	input1 = l_data[id]; input2 = l_data[id + 1];
	temp = input1;
	comp = (input1 < input2 ^ dir) * 4 + add3;
	input1 = shuffle2(input1, input2, as_uint4(comp));
	input2 = shuffle2(input2, temp, as_uint4(comp));
	VECTOR_SORT(input1, dir);
	VECTOR_SORT(input2, dir);

	/* Store the result to global memory */
	g_data[global_start + get_local_id(0)] = input1;
	g_data[global_start + get_local_id(0) + 1] = input2;
}