#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "vl/mathop.h"
#include "vl/sift.h"
#include "vl/generic.h"
#include "vl/ikmeans.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

#include "base64.h"
#include "my_sift.h"

#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;

// g++ -O2 imgs2sifts_vl.cpp base64.cpp -I /Users/agorodilov/work/msr_image/vlfeat-0.9.16/ `pkg-config opencv --cflags --libs` /Users/agorodilov/work/msr_image/vlfeat-0.9.16/bin/maci64/libvl.dylib  -o imgs2sifts_vl

int main() {

    int err = 0 ;

    vl_uint     *asgn = 0;
    vl_ikm_acc  *centers = (vl_ikm_acc*)malloc(sizeof(vl_ikm_acc) * 128 * 8000);

    {
        std::ifstream file("centers.bin", std::ios::binary);
        file.read((char*)centers, sizeof(vl_ikm_acc) * 128 * 8000);
    }

    VlIKMFilt *ikmf;
    vl_ikm_init(ikmf, centers, 128, 8000);

    std::string str;
    double* TFrames = (double*)calloc(4 * 10000, sizeof(double));
    vl_uint8* TDescr  = (vl_uint8*)calloc(128 * 10000, sizeof(vl_uint8));

    int NumImagesProcessed = 0;
	while (!getline(std::cin, str).fail()) {
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

		cv::Mat imgCv = cv::imdecode(imgBin, CV_LOAD_IMAGE_GRAYSCALE);

        int                Tnframes = 0;
        VLSIFT(&imgCv, TDescr, TFrames, &Tnframes, 0);

		NumImagesProcessed ++;

		if (NumImagesProcessed % 1000 == 0) cerr << "processed " << NumImagesProcessed << endl;
	}
}
