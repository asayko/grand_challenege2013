// the program reads from stdin tab-separeted pairs ImgId"\t"Base64EncodedJpegImg
// and writes to stdout:
// ImgId"\t"NumOfSifts"\n"
// SIFT1"\n"
// ...
// SIFT1"\n"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>

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

	/*
	binData.clear();
	boost::archive::iterators::detail::to_6_bit<char> t;
	for (size_t i = 0; i < str64.size(); ++i) {
		std::cerr << t(str64[i]) == -1 << "\n";
	}
	std::cerr << std::endl;

	typedef boost::archive::iterators::transform_width<
	    boost::archive::iterators::binary_from_base64<char *>, 8, 6> TToBinnaryTransformerIter;

	std::copy(
			TToBinnaryTransformerIter(&*str64.begin()),
			TToBinnaryTransformerIter(&*str64.end()),
			std::back_inserter(binData));
	*/
}

int main() {

	std::string str;
	while (!std::getline(std::cin, str).fail()) {

		boost::char_separator<char> sep("\t"); // default constructed
		typedef boost::tokenizer<boost::char_separator<char> > TTok;
		TTok tok(str, sep);
		std::vector<std::string> strs(tok.begin(), tok.end());

		std::string imgId = strs[0];
		boost::algorithm::trim(imgId);
		std::string imgBase64 = strs[1];
		boost::algorithm::trim(imgBase64);

		std::vector<char> imgBin;
		GetBinaryFromBase64(imgBase64, imgBin);

		cv::Mat imgCv = cv::imdecode(imgBin, CV_LOAD_IMAGE_COLOR);

		//
		// cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
		// cv::imshow("Display window", imgCv);
		// cv::waitKey(0);
		//

		cv::Ptr<cv::FeatureDetector> featureDetector = new cv::SiftFeatureDetector(0, 3, 0.08, 5, 1.4);
		std::vector<cv::KeyPoint> keypoints;

		featureDetector->detect(imgCv, keypoints);
		cv::Ptr<cv::SiftDescriptorExtractor> featureExtractor = new cv::SiftDescriptorExtractor();

		cv::Mat descriptors;
		featureExtractor->compute(imgCv, keypoints, descriptors);

		std::cout << imgId << "\t" << descriptors.rows << std::endl;
		for (size_t descIdx = 0; descIdx < descriptors.rows; ++descIdx) {

			cv::Mat tmp;
			cv::normalize(descriptors.row(descIdx), tmp);
			tmp.copyTo(descriptors.row(descIdx));

			for (size_t i = 0; i < descriptors.cols; ++i) {
				std::cout << descriptors.at<float>(descIdx, i) << "\t";
			}
			std::cout << std::endl;
		}
	}

	return 0;
}
