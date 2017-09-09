#include <string>
#include <cstring>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

void anshuReadSparse(string fileName, int *indices, int *markers, unsigned int n, unsigned int bufferlen) {
	std::ifstream file(fileName);
	std::string str;

	int linenum = 0;
	vector<string> list;
	unsigned int totalLen = 0; 
	while (getline(file, str))
	{
		char *mystring = &str[0];
		char * pch;
		pch = strtok(mystring, " ");
		int track = 0;
		list.clear();
		while (pch != NULL)
		{
			if (track % 2 == 1)
				list.push_back(pch);
			track++;
			pch = strtok(NULL, " :");
		}

		markers[linenum] = totalLen;
		for (auto const& var : list) {
			indices[totalLen] = stoi(var);
			totalLen++;
		}
		linenum++;
		if (linenum == n) {
			break;
		}
		if (totalLen >= bufferlen) {
			std::cout << "Buffer too small!" << std::endl;
			markers[linenum] = totalLen; // Final length marker. 
			file.close();
			return;
		}
	}
	markers[linenum] = totalLen; // Final length marker. 
	file.close();

	std::cout << "[anshuReadSparse] Total " << totalLen << " number, " <<
		linenum << " vectors. " << std::endl;
}

/** For reading sparse matrix dataset in index:value format. 

	fileName - name in string
	offset - which datapoint to start reading, normally should be zero
	n - how many data points to read
	indices - array for storing indices
	values - array for storing values
	markers - the start position of each datapoint in indices / values. It have length(n + 1), the last 
	position stores start position of the (n+1)th data point, which does not exist, but convenient 
	for calculating the length of each vector. 
*/

void readSparse(string fileName, int offset, int n, int *indices, float *values, int *markers, unsigned int bufferlen) {
	std::cout << "[readSparse]" << std::endl;

	/* Fill all the markers with the maximum index for the data, to prevent 
	   indexing outside of the range. */
	for (int i = 0; i <= n; i++) {
		markers[i] = bufferlen - 1; 
	}

	std::ifstream file(fileName);
	std::string str;

	unsigned int ct = 0;		// Counting the input vectors. 
	unsigned int totalLen = 0;	// Counting all the elements. 
	while (std::getline(file, str)) // Get one vector (one vector per line). 
	{
		if (ct < offset) {		// If reading with an offset, skip < offset vectors. 
			ct++;
			continue;
		}
		// Constructs an istringstream object iss with a copy of str as content.
		std::istringstream iss(str); 
		// Removes label. 
		std::string sub;
		iss >> sub;
		// Mark the start location. 
		markers[ct - offset] = min(totalLen, bufferlen-1);
		int pos;
		float val;
		int curLen = 0; // Counting elements of the current vector. 
		do
		{
			std::string sub;
			iss >> sub;
			pos = sub.find_first_of(":");
			if (pos == string::npos) {
				continue;
			}
			val = stof(sub.substr(pos + 1, (str.length() - 1 - pos)));
			pos = stoi(sub.substr(0, pos));

			if (totalLen < bufferlen) {
				indices[totalLen] = pos;
				values[totalLen] = val;
			}
			else {
				std::cout << "[readSparse] Buffer is too small, data is truncated!\n";
				return;
			}
			curLen++;
			totalLen++;
		} while (iss);

		ct++;
		if (ct == (offset + n)) {
			break;
		}
	}
	markers[ct - offset] = totalLen; // Final length marker. 
	std::cout << "[readSparse] Read " << totalLen << " numbers, " << 
		ct - offset  << " vectors. " << std::endl;
}

/* Functions for reading and parsing MNIST 60k/10k dataset. 
	url: http://yann.lecun.com/exdb/mnist/
	Reference: https://compvisionlab.wordpress.com/2014/01/01/c-code-for-reading-mnist-data-set/
*/

/* Reversing the endianess of an integer. */
int reverseInt(int i) { 
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void Read_MNIST(string fileName, int NumberOfImages, int SizeDataOfAnImage, float *data) {
	ifstream file(fileName, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = reverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = reverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = reverseInt(n_cols);
		for (int i = 0; i<number_of_images; ++i)
		{
			for (int r = 0; r<n_rows; ++r)
			{
				for (int c = 0; c<n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					// arr[i][(n_rows*r) + c] = (double)temp;
					data[i * SizeDataOfAnImage + n_rows * r + c] = (float)temp;
				}
			}
		}
	}
}

// Reference: http://eric-yuan.me/cpp-read-mnist/
void Read_MNISTLabel(string fileName, int numToRead, int *labels) {
	ifstream file(fileName, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = reverseInt(number_of_images);
		for (int i = 0; i < number_of_images; ++i)
		{
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			labels[i] = (int)temp;
		}
	}
} 

/* Functions for reading and parsing the SIFT dataset. */

void fvecs_read(const std::string& file, int offset, int readsize, float *out) {
	int dimension = 128;

	/* According to sift1b hosting website, 4+d bytes exists for each vector, where d is the dimension. */

	std::ifstream myFile(file, std::ios::in | std::ios::binary);
	long pointer_offset = (long)offset * ((long)4 * dimension + 4);

	if (!myFile) {
		printf("Error opening file ... \n");
		return;
	}
	myFile.seekg(pointer_offset);

	float x[1];
	int ct = 0;
	while (!myFile.eof()) {
		// Dummy reads, 128 0 0 0 separates each vector. 
		myFile.read((char*)x, 4);

		for (int d = 0; d < 128; d++) {
			if (!myFile.read((char*)x, 4)) {
				printf("Error reading file ... \n");
				return;
			}
			out[ct * dimension + d] = (float)*x;
		}

		ct++;
		if (ct == readsize) {
			break;
		}
	}
	myFile.close();
}
/*

Reads the SIFT1B dataset in bvecs format (dimension 128)

Offset and readsize is in unit of datapoints

"queries.bvecs"

*/
void bvecs_read(const std::string& file, int offset, int readsize, float *out) {

	int dimension = 128;

	/* According to sift1b hosting website, 4+d bytes exists for each vector, where d is the dimension. */

	std::ifstream myFile(file, std::ios::in | std::ios::binary);
	long pointer_offset = (long)offset * ((long)dimension + 4);

	if (!myFile) {
		printf("Error opening file ... \n");
		return;
	}
	myFile.seekg(pointer_offset);

	unsigned char x[1];
	int ct = 0;
	while (!myFile.eof()) {
		// Dummy reads, 128 0 0 0 separates each vector. 
		myFile.read((char*)x, 1);
		myFile.read((char*)x, 1);
		myFile.read((char*)x, 1);
		myFile.read((char*)x, 1);

		for (int d = 0; d < 128; d++) {
			if (!myFile.read((char*)x, 1)) {
				printf("Error reading file ... \n");
				return;
			}
			out[ct * dimension + d] = (float)*x;
		}

		ct++;
		if (ct == readsize) {
			break;
		}
	}
	myFile.close();
}

/*

For reading the indices of the groudtruths.
Each indice represent a datapoint in the same order as the base dataset. 

Vector indexing: 
The k_th neighbor of the q_th query is out[(q * availableTopK)]

file - filename
numQueries - the number of query data points
availableTopK - the topk groundtruth available for each vector
out - output vector

*/
void readGroundTruthInt(const std::string& file, int numQueries, int availableTopK, unsigned int *out) {
	std::ifstream myFile(file, std::ios::in | std::ios::binary);

	if (!myFile) {
		printf("Error opening file ... \n");
		return;
	}

	char cNum[256];
	int ct = 0;
	while (myFile.good() && ct < availableTopK * numQueries) {
		myFile.good();
		myFile.getline(cNum, 256, ' ');
		out[ct] = atoi(cNum);
		ct++;
	}

	myFile.close();
}

/*

For reading the distances of the groudtruths.
Each distances represent the distance of the respective base vector in the "indices" to the query.

Vector indexing:
The k_th neighbor's distance to the q_th query is out[(q * availableTopK) + k]

file - filename
numQueries - the number of query data points
availableTopK - the topk groundtruth available for each vector
out - output vector

*/
void readGroundTruthFloat(const std::string& file, int numQueries, int availableTopK, float *out) {
	std::ifstream myFile(file, std::ios::in | std::ios::binary);

	if (!myFile) {
		printf("Error opening file ... \n");
		return;
	}

	char cNum[256];
	int ct = 0;
	while (myFile.good() && ct < availableTopK * numQueries) {
		myFile.good();
		myFile.getline(cNum, 256, ' ');
		out[ct] = strtof(cNum, NULL);
		ct++;
	}

	myFile.close();
}