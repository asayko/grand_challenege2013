#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vl/mathop.h>
#include <vl/sift.h>
#include <vl/generic.h>
#include <vl/ikmeans.h>

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

// g++ -O2 imgs2sifts_vl.cpp base64.cpp my_sift.cpp -I /Users/agorodilov/work/msr_image/vlfeat-0.9.16/ `pkg-config opencv --cflags --libs` /Users/agorodilov/work/msr_image/vlfeat-0.9.16/bin/maci64/libvl.dylib  -o imgs2sifts_vl

void make_clustering(vl_uint8 *data, int N, int dim, int K)
{
    int err = 0 ;

    vl_uint     *asgn = 0 ;
    vl_ikm_acc  *centers = (vl_ikm_acc*)malloc(sizeof(vl_ikm_acc) * dim * K);

    int method_type = VL_IKM_ELKAN;
    int max_niters  = 200 ;
    int verb = 0 ;

    VlIKMFilt *ikmf =  vl_ikm_new (method_type);

    vl_ikm_set_verbosity(ikmf, verb);
    vl_ikm_set_max_niters(ikmf, max_niters);
    vl_ikm_init_rand_data(ikmf, data, dim, N, K);

    err = vl_ikm_train (ikmf, data, N) ;
    if (err) printf("ikmeans: possible overflow!") ;

    memcpy(centers, vl_ikm_get_centers (ikmf), sizeof(vl_ikm_acc) * dim * K);

    {
        std::ofstream file("centers.bin", std::ios::binary);
        file.write((const char*)centers, sizeof(vl_ikm_acc) * dim * K);
    }
}

int main() {
    int MaxNumberOfDescr = 10 * 1024 * 1024;
    vl_uint8* descr = (vl_uint8*)calloc(128 * MaxNumberOfDescr, sizeof(vl_uint8));
    int ndescr = 0;

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

        if (ndescr + Tnframes > MaxNumberOfDescr) {
            continue;
        }

        memcpy(descr + ndescr, TDescr, 128 * Tnframes * sizeof(vl_uint8));
        ndescr += Tnframes;
/*
        char buffer[128*2 + 1];
        buffer[128*2] = 0;
        for(int i = 0; i < Tnframes; i++) {
            for(int j = 0; j < 128; j++)
                sprintf(buffer + 2*j, "%02X", TDescr[i*128 + j]);
            cout << buffer << "\t";
        }
        cout << endl;
*/
        /*
        for(int i = 0; i < Tnframes; i++) {
            circle(imgCv,
            cvPoint(TFrames[0+i*4], TFrames[1+i*4]), TFrames[2+i*4],
            cvScalar(255, 0, 0, 0),
            1, 8, 0);
        }

        cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
		cv::imshow("Display window", imgCv);
		cv::waitKey(0);
		*/

		NumImagesProcessed ++;

		if (NumImagesProcessed % 1000 == 0) cerr << "processed " << NumImagesProcessed << endl;
	}

	cout << "Extracted " << ndescr << " descriptors" << endl;

	make_clustering(descr, ndescr, 128, 8000);
}
