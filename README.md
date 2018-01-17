# FLASH

FLASH (Fast LSH Algorithm for Similarity Search Accelerated with HPC) is a library for large scale approximate nearest neighbor search of sparse vectors. It is currently available in C++ for CPU parallel computing and supports OpenCL enabled GPGPU computing. See [our paper](https://arxiv.org/pdf/1709.01190.pdf) for theoretical and benchmarking details. 

**Coming soon: ** Full GPU ANNS over sparse datasets, providing additional speed up over the current benchmark. 

## Performance

We tested our system on a few large scale sparse datasets including [url](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#url), [webspam](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#webspam) and [kdd12](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#kdd2012). 

### Quality Metrtics

*R@k* is the recall of the 1-nearest neighbor in the top-k results. 
*S@k* is the average cosine similarity of the top-k results concerning the query datapoint. 

For the testing of the sparse datasets, we present results on 2 CPUs (Intel Xeon E5 2660v4) and 2 CPUs + 1 GPU. The following results are from a head-to-head comparison with [NMSLIB](https://github.com/searchivarius/nmslib) v1.6 hnsw, one of the best methods available (see [ann-benchmarks](https://github.com/erikbern/ann-benchmarks)). In particular, we compared the timing for the construction of full knn-graph from grounds up, and the per-query timing (after building the index). We also estimated and compared the memory consumption of the index. 


**Webspam, Url**

<img src="https://github.com/RUSH-LAB/Flash/blob/master/plots/webspam_url_table.PNG" width="668" height="85" />
<img src="https://github.com/RUSH-LAB/Flash/blob/master/plots/webspam_plots.PNG" width="739" height="310" />
<img src="https://github.com/RUSH-LAB/Flash/blob/master/plots/url_plots.PNG" width="739" height="310" />

**Kdd2012**

<img src="https://github.com/RUSH-LAB/Flash/blob/master/plots/kdd12_table.PNG" width="530" height="58" />

## Prerequisites

The current version of the software is tested on 64-bit machines running Ubuntu 16.04, with CPU and at least 1 GPGPU installed. The compiler needs to support C++11 and OpenMP. GPGPU support of OpenCL 1.1 or OpenCL 2.0 is required. For example, OpenCL on Nvidia graphics cards requires the installation of [CUDA](https://developer.nvidia.com/cuda-toolkit-32-downloads). 

## System Configuration

Navigate to the FLASH directory. First configure the system by editing the following section of *LSHReservoir_config.h* to choose the devices to use. 

```
// Customize processing architecture. 
#define OPENCL_HASHTABLE // Placing the hashtable in the OpenCL device. 
#define OPENCL_HASHING   // Perform hashing in the OpenCL device. 
#define OPENCL_KSELECT	 // Perform k-selection in the OpenCL device. 

// Comment out if using OpenCL 1.XX. 
#define OPENCL_2XX

// Select the id of the desired platform and device, only relevant when using OpenCl. 
// An overview of the platforms and devices can be queried through the OpenCL framework. 
// On Linux, a package "clinfo" is also capable of outputing the platform and device information. 
#define CL_PLATFORM_ID 0
#define CL_DEVICE_ID 0
```
For dense datasets:
1. GPU only

```
// Customize processing architecture. 
#define OPENCL_HASHTABLE // Placing the hashtable in the OpenCL device. 
#define OPENCL_HASHING   // Perform hashing in the OpenCL device. 
#define OPENCL_KSELECT	 // Perform k-selection in the OpenCL device. 
```
For sparse datasets:
1. CPU + GPU

```
// Customize processing architecture. 
//#define OPENCL_HASHTABLE // Placing the hashtable in the OpenCL device. 
//#define OPENCL_HASHING   // Perform hashing in the OpenCL device. 
#define OPENCL_KSELECT	 // Perform k-selection in the OpenCL device. 
```
2. CPU only
```
// Customize processing architecture. 
//#define OPENCL_HASHTABLE // Placing the hashtable in the OpenCL device. 
//#define OPENCL_HASHING   // Perform hashing in the OpenCL device. 
//#define OPENCL_KSELECT	 // Perform k-selection in the OpenCL device. 
```
Install clinfo by `apt-get install clinfo`. Fill in CL_PLATFORM_ID / CL_DEVICE_ID to choose the desired platform and device based on to the order that the GPU platforms and devices appear in the output of clinfo. Comment out `OPENCL_2XX` if using OpenCL 1.X. Save and close the file. 

Complete the dataset setup as detailed in the **Tutorial** section (or any customized usage, please refer to our [documentation](https://github.com/RUSH-LAB/Flash/blob/master/doc.pdf)), and compile the program: 
```
make clean; make
```
The compilation is complete if no errors appear. Run the program by: 
```
./runme
```

## Tutorial

We will present very detailed steps to replicate one result presented in [our paper](https://arxiv.org/pdf/1709.01190.pdf), in particular the webspam dataset. Other results can be replicated in a very similar manner. For customized usage, please refer to our [documentation](https://github.com/RUSH-LAB/Flash/blob/master/doc.pdf) generated by [doxygen](http://www.stack.nl/~dimitri/doxygen/). 

Download the dataset from [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#webspam), and the groundtruths from  this [link](https://github.com/wangyiqiu/webspam). Place the dataset and groundtruth files in a directory you like. 

Open `benchmarking.h` and follow the following configuration. Make sure to set the **path** of the dataset and groundtruth files correctly under `#elif defined WEBSPAM_TRI`. 

```
/* Select a dataset below by uncommenting it. 
Then modify the file location and parameters below in the Parameters section. */

//#define URL
#define WEBSPAM_TRI
//#define KDD12

...

#elif defined WEBSPAM_TRI

...

#define BASEFILE		".../trigram.svm"
#define QUERYFILE		".../trigram.svm"
#define GTRUTHINDICE		".../webspam_tri_gtruth_indices.txt"
#define GTRUTHDIST		".../webspam_tri_gtruth_distances.txt"
```
Configure the system for **sparse data**, CPU or CPU + GPU and run the program (if not already done, see **System Configuration** above). 

The test program builds multiple hash tables for the dataset and query 10,000 test vectors followed by quality evaluations. The program will run with console outputs, indicating the progress and performance. `make clean; make` is required after changing the parameters. Please note that the time for parsing the webspam dataset from disk might take about 5-10 minutes. 

## Authors

- [Yiqiu Wang](https://github.com/wangyiqiu)
- [Anshumali Shrivastava](https://www.cs.rice.edu/~as143/)
- [Jonathan Wang](https://www.linkedin.com/in/jonathan-wang-725ab28a/)
- [Heejung Ryu](https://github.com/bluejay9676)

## License

This library is licensed under Apache-2.0. See LICENSE for more details. 

## Acknowledgments

* Rice University Sketching and Hashing Lab ([RUSH Lab](http://rush.rice.edu/index.html)) provided the computing platform for testing. 
