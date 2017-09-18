#pragma once

// Comment out if not using GPU. 
//#define USE_OPENCL

// Comment out if using OpenCL 1.XX. Does not matter if not usng GPU. 
#define OPENCL_2XX

// Select the id of the desired platform and device, only relevant when using OpenCl. 
// An overview of the platforms and devices can be queried through the OpenCL framework. 
// On Linux, a package "clinfo" is also capable of outputing the platform and device information. 
#define CL_PLATFORM_ID 0
#define CL_DEVICE_ID 0

#if defined USE_OPENCL
// The parts of the program using OpenCL can be customized. 
#define OPENCL_HASHTABLE
//#define CPU_TB
#define OPENCL_HASHING
//#define CPU_HASHING
#define OPENCL_KSELECT
//#define CPU_KSELECT
#else
#define CPU_TB
#define CPU_HASHING
#define CPU_KSELECT
#endif

/* Performance tuning params, Do not touch. */
#define wg_segSize 512			// Number of workgroup element, an integral factor of the segmentSize. 
#define l_segSize 64			// Number of elements each thread will tally. 
//#define SECONDARY_HASHING
