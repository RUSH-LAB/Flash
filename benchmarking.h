#pragma once

/* Select a dataset below by uncommenting it. 
Then modify the file location and parameters below in the Parameters section. */

//#define SIFT1M
#define URL
//#define WEBSPAM_TRI
//#define KDD12

/* Parameters. */

#if defined SIFT1M

#define DENSE_DATASET

#define NUMHASHBATCH				100
#define BATCHPRINT					10

#define RANGE_POW					22
#define RANGE_ROW_U					18
#define SAMFACTOR					24
#define NUMTABLES					512
#define RESERVOIR_SIZE				32
#define OCCUPANCY					0.4

#define QUERYPROBES					1
#define HASHINGPROBES				1

#define DIMENSION					128
#define FULL_DIMENSION				128
#define NUMQUERY					10000
#define NUMBASE						1000000
#define MAX_RESERVOIR_RAND			100000

#define AVAILABLE_TOPK				1000
#define TOPK						128

#define BASEFILE		"../files/datasets/sift1m/sift_base.fvecs"
#define QUERYFILE		"../files/datasets/sift1m/sift_query.fvecs"
#define GTRUTHINDICE	"../files/datasets/sift1m/sift1m_gtruth_indices.txt"
#define GTRUTHDIST		"../files/datasets/sift1m/sift1m_gtruth_distances.txt"

#elif defined URL

#define SPARSE_DATASET

#define NUMHASHBATCH				200
#define BATCHPRINT					10

#define K							4
#define RANGE_POW					15
#define RANGE_ROW_U					15

#define NUMTABLES					128
#define RESERVOIR_SIZE				32
#define OCCUPANCY					1

#define QUERYPROBES					1
#define HASHINGPROBES				1

#define DIMENSION					120
#define FULL_DIMENSION				3231961
#define NUMBASE						2386130
#define MAX_RESERVOIR_RAND			2386130
#define NUMQUERY					10000
#define TOPK						128
#define AVAILABLE_TOPK				1024

#define NUMQUERY					10000
#define AVAILABLE_TOPK				1024
#define TOPK						128

#define BASEFILE		"../files/datasets/url/url_combined"
#define QUERYFILE		"../files/datasets/url/url_combined"
#define GTRUTHINDICE	"../files/datasets/url/url_gtruth_indices.txt"
#define GTRUTHDIST		"../files/datasets/url/url_gtruth_distances.txt"

#elif defined WEBSPAM_TRI

#define SPARSE_DATASET

#define NUMHASHBATCH				50
#define BATCHPRINT					5

#define K							4
#define RANGE_POW					15
#define RANGE_ROW_U					15

#define NUMTABLES					32
#define RESERVOIR_SIZE				64
#define OCCUPANCY					1

#define QUERYPROBES					1
#define HASHINGPROBES				1

#define DIMENSION					4000
#define FULL_DIMENSION				16609143
#define NUMBASE						340000
#define MAX_RESERVOIR_RAND			35000
#define NUMQUERY					10000
#define TOPK						128
#define AVAILABLE_TOPK				1024

#define NUMQUERY					10000
#define AVAILABLE_TOPK				1024
#define TOPK						128

#define BASEFILE		"../files/datasets/webspam/trigram.svm"
#define QUERYFILE		"../files/datasets/webspam/trigram.svm"
#define GTRUTHINDICE	"../files/datasets/webspam/webspam_tri_gtruth_indices.txt"
#define GTRUTHDIST		"../files/datasets/webspam/webspam_tri_gtruth_distances.txt"

#elif defined KDD12

#define SPARSE_DATASET

#define NUMHASHBATCH				20000
#define BATCHPRINT					2000

#define K							4
#define RANGE_POW					20
#define RANGE_ROW_U					20

#define NUMTABLES					8
#define RESERVOIR_SIZE				64
#define OCCUPANCY					1

#define QUERYPROBES					1
#define HASHINGPROBES				1

#define DIMENSION					12
#define FULL_DIMENSION				54686452
#define NUMBASE						149629105
#define MAX_RESERVOIR_RAND			149629105
#define NUMQUERY					10000
#define TOPK						128
#define AVAILABLE_TOPK				1024

#define BASEFILE		"../files/datasets/kdd2012/kdd12"
#define QUERYFILE		"../files/datasets/kdd2012/kdd12"
#define GTRUTHINDICE	"../files/datasets/kdd2012/kdd12_gtruth_indices.txt"
#define GTRUTHDIST		"../files/datasets/kdd2012/kdd12_gtruth_distances.txt"

#elif defined FRIENDSTER

#define SPARSE_DATASET

#define NUMHASHBATCH				10000
#define BATCHPRINT					500

#define K							2
#define RANGE_POW					20
#define RANGE_ROW_U					20

#define NUMTABLES					32
#define RESERVOIR_SIZE				128
#define OCCUPANCY					1

#define QUERYPROBES					1
#define HASHINGPROBES				1

#define DIMENSION					30
#define FULL_DIMENSION				65608366
#define NUMBASE						65598366
#define MAX_RESERVOIR_RAND			65608366
#define NUMQUERY					10000
#define TOPK						128
#define AVAILABLE_TOPK				1024

#define BASEFILE		"../files/datasets/friendster/friendster.svm"
#define QUERYFILE		"../files/datasets/friendster/friendster.svm"
#define GTRUTHINDICE	"../files/datasets/friendster/friendster_gtruth_indices.txt"
#define GTRUTHDIST		"../files/datasets/friendster/friendster_gtruth_distances.txt"

#elif defined WEBSPAM_UNI

#define SPARSE_DATASET

#define NUMHASHBATCH				10
#define BATCHPRINT					2

#define K							4
#define RANGE_POW					12
#define RANGE_ROW_U					12

#define NUMTABLES					32
#define RESERVOIR_SIZE				64
#define OCCUPANCY					1

#define QUERYPROBES					1
#define HASHINGPROBES				1

#define DIMENSION					254
#define FULL_DIMENSION				254
#define NUMBASE						10000
#define MAX_RESERVOIR_RAND			10000
#define NUMQUERY					100
#define TOPK						128
#define AVAILABLE_TOPK				128

#define BASEFILE		"../files/datasets/webspam/unigram.svm"
#define QUERYFILE		"../files/datasets/webspam/unigram.svm"
#define GTRUTHINDICE	"../files/datasets/webspam_uni_gtruth_indices.txt"
#define GTRUTHDIST		"../files/datasets/webspam_uni_gtruth_distances.txt"

#elif defined(SIFT10M)

#define DENSE_DATASET

#define NUMHASHBATCH				1000
#define BATCHPRINT					100

#define RANGE_POW					22
#define RANGE_ROW_U					18
#define SAMFACTOR					24
#define NUMTABLES					512
#define RESERVOIR_SIZE				32
#define OCCUPANCY					0.4

#define QUERYPROBES					1
#define HASHINGPROBES				1

#define DIMENSION					128
#define FULL_DIMENSION				128
#define NUMQUERY					10000
#define NUMBASE						10000000
#define MAX_RESERVOIR_RAND			100000

#define AVAILABLE_TOPK				1000
#define TOPK						128

#define BASEFILE "../files/datasets/sift1b/bigann_base.bvecs"
#define QUERYFILE "../files/datasets/sift1b/bigann_query.bvecs"
#define GTRUTHINDICE "../files/datasets/sift1b/sift10m_gtruth_indices.txt"
#define GTRUTHDIST "../files/datasets/sift1b/sift10m_gtruth_distances.txt"

#elif defined SIFT1B

#define DENSE_DATASET

#define NUMHASHBATCH				100000
#define BATCHPRINT					10000

#define RANGE_POW					25
#define RANGE_ROW_U					22
#define SAMFACTOR					24
#define NUMTABLES					512
#define RESERVOIR_SIZE				32
#define OCCUPANCY					0.4

#define QUERYPROBES					1
#define HASHINGPROBES				1

#define DIMENSION					128
#define FULL_DIMENSION				128
#define NUMQUERY					10000
#define NUMBASE						10000000
#define MAX_RESERVOIR_RAND			100000

#define AVAILABLE_TOPK				100
#define TOPK						64

#define BASEFILE		"../files/datasets/sift1b/bigann_base.bvecs"
#define QUERYFILE		"../files/datasets/sift1b/bigann_query.bvecs"
#define GTRUTHINDICE	"../files/datasets/sift1b/sift1b_gtruth_indices.txt"
#define GTRUTHDIST		"../files/datasets/sift1b/sift1b_gtruth_distances.txt"

#elif defined SIFTSMALLTEST

#define DENSE_DATASET

#define NUMHASHBATCH				10
#define BATCHPRINT					5
#define RANGE_POW					12 
#define RANGE_ROW_U					12
#define RESERVOIR_SIZE				16 
#define NUMTABLES					32 
 
#define DIMENSION					128 
#define FULL_DIMENSION				128
#define NUMBASE						10000 
#define MAX_RESERVOIR_RAND			10000 
#define QUERYPROBES					2
#define HASHINGPROBES				1 
#define SAMFACTOR					24 
#define OCCUPANCY					0.4 

#define NUMQUERY					100
#define AVAILABLE_TOPK				100
#define TOPK						64

#define BASEFILE		"siftsmall_base.fvecs"
#define QUERYFILE		"siftsmall_query.fvecs"
#define GTRUTHINDICE	"siftsmall_gtruth_indices.txt"
#define GTRUTHDIST		"siftsmall_gtruth_distances.txt"
#endif

void benchmark_kselect();
void benchmark_naiverp(int RANDPROJ_COMPRESS);
void benchmark_paragrid();
void benchmark_bruteforce();
void benchmark_ava();
void benchmark_friendster_quality();
void benchmark_sparse();
void benchmark_dense();
void benchmark_doph(int TEST_DOPH);
void benchmark_smartrp(int SMART_RP);

#if !defined (DENSE_DATASET)
#define SAMFACTOR 24 // DUMMY. 
#endif

#if !defined (SPARSE_DATASET)
#define K 10 // DUMMY
#endif

//#define FRIENDSTER
//#define SIFTSMALLTEST
//#define SIFT10M
//#define SIFT1B
