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

void ExtractVisWordsToFout(
		const cv::Mat & imgCv,
		std::ofstream * ExtractedVisWordsOutFile,
		pthread_mutex_t * ExtractedVisWordsOutFileLock,
		cv::flann::Index * VisualVocabularyIndex,
		long unsigned int * processedImagesCounter,
		const std::string & imgId) {

	cv::Ptr<cv::FeatureDetector> featureDetector = new cv::SiftFeatureDetector(0, 3, 0.08, 5, 1.4);
	std::vector<cv::KeyPoint> keypoints;

	featureDetector->detect(imgCv, keypoints);
	cv::Ptr<cv::SiftDescriptorExtractor> featureExtractor = new cv::SiftDescriptorExtractor();

	cv::Mat descriptors;
	featureExtractor->compute(imgCv, keypoints, descriptors);

	for (size_t descIdx = 0; descIdx < descriptors.rows; ++descIdx) {
		cv::Mat tmp;
		cv::normalize(descriptors.row(descIdx), tmp);
		tmp.copyTo(descriptors.row(descIdx));
	}

	cv::Mat indices(descriptors.rows, 1, CV_32S);
	cv::Mat dists(descriptors.rows, 1, CV_32F);

	if (descriptors.rows > 0) {
		VisualVocabularyIndex->knnSearch(descriptors, indices, dists, 1);
	}

	// Get lock
	pthread_mutex_lock(ExtractedVisWordsOutFileLock);

	(*ExtractedVisWordsOutFile) << imgId << "\t";
	for (size_t visWordIdx = 0; visWordIdx < indices.rows; ++visWordIdx) {
		(*ExtractedVisWordsOutFile) << indices.at<int>(visWordIdx, 0) << " ";
	}
	(*ExtractedVisWordsOutFile) << std::endl;

	//if (*processedImagesCounter % 1000 == 0) {
		std::cerr << *processedImagesCounter << " images proceed." << std::endl;
	//}

	++(*processedImagesCounter);
	// Release lock
	pthread_mutex_unlock(ExtractedVisWordsOutFileLock);
}


struct TVisWordsExtractorThreadParams {
	const char * FileName;
	size_t FileSize;
	size_t ThreadNum;
	size_t ThreadsNum;
	cv::flann::Index * VisualVocabularyIndex;
	pthread_mutex_t * ExtractedVisWordsOutFileLock;
	std::ofstream * ExtractedVisWordsOutFile;
	long unsigned int  * ProcessedImagesCounter;
};

void * ExtractVisWordsThreadFunc(void * _params) {
	const TVisWordsExtractorThreadParams * params = (const TVisWordsExtractorThreadParams *) _params;
	std::ifstream fin(params->FileName);
	fin.seekg(params->ThreadNum * (params->FileSize / params->ThreadsNum));

	size_t lastByte = params->FileSize;
	if (params->ThreadNum != params->ThreadsNum - 1) {
		lastByte = (params->ThreadNum + 1) * (params->FileSize / params->ThreadsNum) + 1;
	}

	if (params->ThreadNum != 0) {
		while (!fin.fail() &&  fin.get() != '\n');
	}

	while (!fin.fail() && fin.tellg() < lastByte) {
		std::string str;
		std::getline(fin, str);

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

			ExtractVisWordsToFout(imgCv,
					params->ExtractedVisWordsOutFile,
					params->ExtractedVisWordsOutFileLock,
					params->VisualVocabularyIndex,
					params->ProcessedImagesCounter,
					imgId);

		} catch (...) {
			std::cerr << "Error while processing " << imgId << std::endl;
		}
	}

	return NULL;
}

void WriteCvMatToFile(const cv::Mat & mat, const char * fileName) {
	std::ofstream fout(fileName);

	fout << mat.rows << "\t" << mat.cols << std::endl;
	for (size_t rowId = 0; rowId < mat.rows; ++rowId) {
		for (size_t colId = 0; colId < mat.cols; ++colId) {
			fout << mat.at<float>(rowId , colId) << "\t";
		}
		fout << std::endl;
	}

}

void ReadCvMatFromFile(cv::Mat & mat, const char * fileName) {
	std::ifstream fin(fileName);
	size_t rows;
	size_t cols;

	fin >> rows;
	fin >> cols;

	mat.create(rows, cols, CV_32F);

	for (size_t rowId = 0; rowId < rows; ++rowId) {
		for (size_t colId = 0; colId < cols; ++colId) {
			fin >> mat.at<float>(rowId, colId);
		}
	}

	return;
}

int main() {
	const char * fileName = "/Users/asayko/data/grand_challenge/Train/TrainImageSetSmall.tsv";
	const char * inVocabularyFileName = "vocabulary10000.tsv";
	const char * outVisWordsFileName = "vis_words_10000.tsv";

	std::ifstream fin(fileName, std::ifstream::in | std::ifstream::binary);
	fin.seekg(0, std::ifstream::end);
	size_t fileSize = fin.tellg();
	const size_t NUM_THREADS = 2;

	cv::Mat visualVocabularyMat;
	ReadCvMatFromFile(visualVocabularyMat, inVocabularyFileName);
	cv::flann::Index visualVocabularyIndex(visualVocabularyMat, cv::flann::LinearIndexParams());

	std::ofstream visWordsFout(outVisWordsFileName);
	pthread_mutex_t visWordsFoutLock;
	long unsigned int processedImagesCounter = 0;

	pthread_mutex_init(&visWordsFoutLock, NULL);

	pthread_t threads[NUM_THREADS];
	TVisWordsExtractorThreadParams visWordsExtractorThreadParams[NUM_THREADS];
	for (size_t i = 0; i < NUM_THREADS; ++i) {
		visWordsExtractorThreadParams[i].FileName = fileName;
		visWordsExtractorThreadParams[i].FileSize = fileSize;
		visWordsExtractorThreadParams[i].ThreadNum = i;
		visWordsExtractorThreadParams[i].ThreadsNum = NUM_THREADS;
		visWordsExtractorThreadParams[i].VisualVocabularyIndex = &visualVocabularyIndex;
		visWordsExtractorThreadParams[i].ExtractedVisWordsOutFile = &visWordsFout;
		visWordsExtractorThreadParams[i].ExtractedVisWordsOutFileLock = &visWordsFoutLock;
		visWordsExtractorThreadParams[i].ProcessedImagesCounter = &processedImagesCounter;

		int rc = pthread_create(&threads[i],
		                        NULL,
		                        ExtractVisWordsThreadFunc,
		                        (void *) &visWordsExtractorThreadParams[i]);
		if (rc) {
			std::cerr << "ERROR; return code from pthread_create() is " << rc << std::endl;
			exit(1);
		}
	}

    for (size_t i = 0; i < NUM_THREADS; ++i) {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&visWordsFoutLock);
    visWordsFout.close();
}
