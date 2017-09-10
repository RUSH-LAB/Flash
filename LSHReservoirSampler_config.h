#pragma once

// Comment out if not using GPU. 
#define USE_GPU
// Comment out if using OpenCL 1.XX. Does not matter if not usng GPU. 
#define OPENCL_2XX

#define CL_GPU_PLATFORM 0 // Does not matter if not usng OpenCL-GPU. 
#define CL_CPU_PLATFORM 1 // Does not matter if not usng OpenCL-CPU. 
#define CL_GPU_DEVICE 0 // Does not matter if not usng OpenCL-GPU. 
#define CL_CPU_DEVICE 0 // Does not matter if not usng OpenCL-CPU. 

// Choose to work with sparse or dense data. 
#define SPARSE
//#define DENSE
/* Performance tuning params, Do not touch. */
#define wg_segSize 512			// Number of workgroup element, an integral factor of the segmentSize. 
#define l_segSize 64			// Number of elements each thread will tally. 
//#define SECONDARY_HASHING
#if defined USE_GPU
//#define GPU_TB
#define CPU_TB
//#define GPU_HASHING
#define CPU_HASHING
#define GPU_KSELECT
//#define CPU_KSELECT
#else
#define CPU_TB
#define CPU_HASHING
#define CPU_KSELECT
#endif

