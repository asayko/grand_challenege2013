#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/flann/flann.hpp>

#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>

#include <pthread.h>

#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cassert>

static const char base64_bkw[] =
"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"
"\0\0\0\0\0\0\0\0\0\0\0\76\0\76\0\77\64\65\66\67\70\71\72\73\74\75\0\0\0\0\0\0"
"\0\0\1\2\3\4\5\6\7\10\11\12\13\14\15\16\17\20\21\22\23\24\25\26\27\30\31\0\0\0\0\77"
"\0\32\33\34\35\36\37\40\41\42\43\44\45\46\47\50\51\52\53\54\55\56\57\60\61\62\63\0\0\0\0\0"
"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"
"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"
"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"
"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";

inline void uudecode_1(char *dst, unsigned char *src) {
    dst[0] = char((base64_bkw[src[0]] << 2) | (base64_bkw[src[1]] >> 4));
    dst[1] = char((base64_bkw[src[1]] << 4) | (base64_bkw[src[2]] >> 2));
    dst[2] = char((base64_bkw[src[2]] << 6) | base64_bkw[src[3]]);
}

size_t Base64Decode(void* dst, const char* b, const char* e) {
    size_t n = 0;

    if ((e - b) % 4) {
        throw std::logic_error("incorrect input length for base64 decode");
    }

    while (b < e) {
        uudecode_1((char*)dst + n, (unsigned char*)b);

        b += 4;
        n += 3;
    }

    if (n > 0) {
        if (b[-1] == ','  || b[-1] == '=') {
            n--;

            if (b[-2] == ',' || b[-2] == '=') {
                n--;
            }
        }
    }
    return n;
}

void GetBinaryFromBase64(const std::string & str64, std::vector<char> & binData) {
	binData.resize(str64.size());
	size_t t = Base64Decode(&*binData.begin(), str64.c_str(), str64.c_str() + str64.size());
	binData.resize(t);
}

void ExtractDescriptorsToStorage(
		const cv::Mat & imgCv,
		cv::Mat & allDescriptors,
		pthread_mutex_t * allDescriptorsLock) {
	cv::Ptr<cv::FeatureDetector> featureDetector = new cv::SiftFeatureDetector(0, 3, 0.08, 5, 1.4);
	std::vector<cv::KeyPoint> keypoints;

	featureDetector->detect(imgCv, keypoints);
	cv::Ptr<cv::SiftDescriptorExtractor> featureExtractor = new cv::SiftDescriptorExtractor();

	cv::Mat descriptors;
	featureExtractor->compute(imgCv, keypoints, descriptors);

	// Get lock
	pthread_mutex_lock(allDescriptorsLock);

	for (size_t descIdx = 0; descIdx < descriptors.rows; ++descIdx) {
		cv::Mat tmp;
		cv::normalize(descriptors.row(descIdx), tmp);
		allDescriptors.push_back(tmp);
	}

	std::cerr << allDescriptors.rows << "descriptors obtained." << std::endl;

	// Release lock
	pthread_mutex_unlock(allDescriptorsLock);
}


struct TSiftExtractorThreadParams {
	const char * FileName;
	size_t FileSize;
	size_t ThreadNum;
	size_t ThreadsNum;
	double ImageSamplingProb;
	pthread_mutex_t * extractedSiftStorageLock;
	cv::Mat * extractedSiftStorage;
};

void * ExtractSiftsThreadFunc(void * _params) {
	const TSiftExtractorThreadParams * params = (const TSiftExtractorThreadParams *) _params;
	std::ifstream fin(params->FileName);
	fin.seekg(params->ThreadNum * (params->FileSize / params->ThreadsNum));

	size_t lastByte = params->FileSize;
	if (params->ThreadNum != params->ThreadsNum - 1) {
		lastByte = (params->ThreadNum + 1) * (params->FileSize / params->ThreadsNum) + 1;
	}

	if (params->ThreadNum != 0) {
		while (!fin.fail() &&  fin.get() != '\n');
	}

	if (params->ThreadNum == 1) std::cerr << "Ola" << std::endl;

	while (!fin.fail() && fin.tellg() < lastByte) {
		std::string str;
		std::getline(fin, str);

		if ((double)rand() / RAND_MAX > params->ImageSamplingProb) {
			continue; // sample data for fast experiments
		}

		boost::char_separator<char> sep("\t");
		typedef boost::tokenizer<boost::char_separator<char> > TTok;
		TTok tok(str, sep);
		std::vector<std::string> strs(tok.begin(), tok.end());

		if (2 != strs.size()) {
			std::cerr << "Bad line for imgId: " << str.substr(0, 10) << std::endl;
		}
		assert(2 == strs.size());

		std::string imgId = strs[0];
		boost::algorithm::trim(imgId);
		std::string imgBase64 = strs[1];
		boost::algorithm::trim(imgBase64);

		try {
			std::vector<char> imgBin;
			GetBinaryFromBase64(imgBase64, imgBin);
			cv::Mat imgCv = cv::imdecode(imgBin, CV_LOAD_IMAGE_COLOR);
			ExtractDescriptorsToStorage(imgCv,
					*params->extractedSiftStorage,
					params->extractedSiftStorageLock);
		} catch (...) {
			std::cerr << "Error while processing " << imgId << std::endl;
		}
	}

	return NULL;
}

int main() {
	const char * fileName = "/Users/asayko/data/grand_challenge/Train/TrainImageSetSmall.tsv";
	const double imageSamplingProb = 0.5;
	std::ifstream fin(fileName, std::ifstream::in | std::ifstream::binary);
	fin.seekg(0, std::ifstream::end);
	size_t fileSize = fin.tellg();
	const size_t NUM_THREADS = 12;

	cv::Mat descriptorsStorage(0, 128, CV_32F);
	descriptorsStorage.reserve(30000000);
	pthread_mutex_t descriptorsStorageLock;
	pthread_mutex_init(&descriptorsStorageLock, NULL);

	pthread_t threads[NUM_THREADS];
	TSiftExtractorThreadParams siftExtractorThreadParams[NUM_THREADS];
	for (size_t i = 0; i < NUM_THREADS; ++i) {
		siftExtractorThreadParams[i].FileName = fileName;
		siftExtractorThreadParams[i].FileSize = fileSize;
		siftExtractorThreadParams[i].ThreadNum = i;
		siftExtractorThreadParams[i].ThreadsNum = NUM_THREADS;
		siftExtractorThreadParams[i].ImageSamplingProb = imageSamplingProb;
		siftExtractorThreadParams[i].extractedSiftStorageLock = &descriptorsStorageLock;
		siftExtractorThreadParams[i].extractedSiftStorage = &descriptorsStorage;

		int rc = pthread_create(&threads[i],
		                        NULL,
		                        ExtractSiftsThreadFunc,
		                        (void *) &siftExtractorThreadParams[i]);
		if (rc) {
			std::cerr << "ERROR; return code from pthread_create() is " << rc << std::endl;
			exit(1);
		}
	}

    for (size_t i = 0; i < NUM_THREADS; ++i) {
        pthread_join(threads[i], NULL);
    }


    pthread_mutex_destroy(&descriptorsStorageLock);
    return 0;
}
