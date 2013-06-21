#include "vl/mathop.h"
#include "vl/sift.h"
#include "vl/generic.h"
#include "vl/ikmeans.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

void VLSIFT(cv::Mat* image, vl_uint8* DATAdescr, double* DATAframes, int* nframes, int verbose );
