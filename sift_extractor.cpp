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


#include <jpeglib.h>

#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/archive/iterators/binary_from_base64.hpp>
#include <boost/archive/iterators/transform_width.hpp>
#include <boost/cstdint.hpp>

#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cassert>

void GetBinaryFromBase64(const std::string & str64, std::vector<char> & binData) {
	binData.clear();

	typedef boost::archive::iterators::transform_width<
	    boost::archive::iterators::binary_from_base64<char *>, 8, 6> TToBinnaryTransformerIter;

	std::copy(
			TToBinnaryTransformerIter(&*str64.begin()),
			TToBinnaryTransformerIter(&*str64.end()),
			std::back_inserter(binData));
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
