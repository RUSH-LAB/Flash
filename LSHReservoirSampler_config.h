#pragma once

//#define USE_GPU

#define SPARSE
//#define DENSE

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

/* OpenCL Platform and Device. */
/* T460 Laptop. */
//#define OPENCL_2XX
//#define VISUAL_STUDIO
//#define CL_GPU_PLATFORM 0
//#define CL_CPU_PLATFORM 1
//#define CL_GPU_DEVICE 0
//#define CL_CPU_DEVICE 0

/* Jaya. */
#define CL_GPU_PLATFORM 1
#define CL_CPU_PLATFORM 0
#define CL_GPU_DEVICE 0
#define CL_CPU_DEVICE 0

/* Performance tuning params, Do not touch. */
#define wg_segSize 512			// Number of workgroup element, an integral factor of the segmentSize. 
#define l_segSize 64			// Number of elements each thread will tally. 
