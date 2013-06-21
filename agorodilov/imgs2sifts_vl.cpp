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

#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;

// g++ -O2 imgs2sifts_vl.cpp -I /Users/agorodilov/work/msr_image/vlfeat-0.9.16/ `pkg-config opencv --cflags --libs` /Users/agorodilov/work/msr_image/vlfeat-0.9.16/bin/maci64/libvl.dylib  -o imgs2sifts_vl

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

VL_INLINE void
transpose_descriptor (vl_sift_pix* dst, vl_sift_pix* src)
{
	int const BO = 8 ;  /* number of orientation bins */
	int const BP = 4 ;  /* number of spatial bins     */
	int i, j, t ;

	for (j = 0 ; j < BP ; ++j) {
		int jp = BP - 1 - j ;
		for (i = 0 ; i < BP ; ++i) {
			int o  = BO * i + BP*BO * j  ;
			int op = BO * i + BP*BO * jp ;
			dst [op] = src[o] ;
			for (t = 1 ; t < BO ; ++t)
				dst [BO - t + op] = src [t + o] ;
		}
	}
}

void VLSIFT(cv::Mat* image, vl_uint8* DATAdescr, double* DATAframes, int* nframes, int verbose = 1){
	//Take IplImage -> convert to SINGLE (float):
	float* frame = (float*)malloc(image->rows * image->cols * sizeof(float));
	uchar* Ldata = (uchar *)image->data;

	for(int i = 0; i < image->rows; i++)
		for(int j = 0; j < image->cols; j++)
			frame[j*image->rows + i] = (float)Ldata[i*image->step + j];

	// VL SIFT computation:
	vl_sift_pix const *data ;
	int                M, N ;
	data = (vl_sift_pix*)frame;
	M = image->rows;
	N = image->cols;

	int                O     =   -1 ; //Octaves
	int                S     =   3 ;  //Levels
	int                o_min =   0 ;

	double             edge_thresh = -1 ;
	double             peak_thresh = -1 ;
	double             norm_thresh = -1 ;
	double             magnif      = -1 ;
	double             window_size = -1 ;

	double            *ikeys = 0 ; //?
	int                nikeys = -1 ; //?
	vl_bool            force_orientations = 0 ;
	vl_bool            floatDescriptors = 0 ;

	/* -----------------------------------------------------------------
	*                                                            Do job
	* -------------------------------------------------------------- */
	{
		VlSiftFilt         *filt ;
		vl_bool            first ;
		double             *frames = 0 ;
		vl_uint8           *descr  = 0 ;
		int                reserved = 0, i,j,q ;

		/* create a filter to process the image */
		filt = vl_sift_new (M, N, O, S, o_min) ;

		if (peak_thresh >= 0) vl_sift_set_peak_thresh (filt, peak_thresh) ;
		if (edge_thresh >= 0) vl_sift_set_edge_thresh (filt, edge_thresh) ;
		if (norm_thresh >= 0) vl_sift_set_norm_thresh (filt, norm_thresh) ;
		if (magnif      >= 0) vl_sift_set_magnif      (filt, magnif) ;
		if (window_size >= 0) vl_sift_set_window_size (filt, window_size) ;

		if (verbose) {
			printf("vl_sift: filter settings:\n") ;
			printf("vl_sift:   octaves      (O)      = %d\n",
				vl_sift_get_noctaves      (filt)) ;
			printf("vl_sift:   levels       (S)      = %d\n",
				vl_sift_get_nlevels       (filt)) ;
			printf("vl_sift:   first octave (o_min)  = %d\n",
				vl_sift_get_octave_first  (filt)) ;
			printf("vl_sift:   edge thresh           = %g\n",
				vl_sift_get_edge_thresh   (filt)) ;
			printf("vl_sift:   peak thresh           = %g\n",
				vl_sift_get_peak_thresh   (filt)) ;
			printf("vl_sift:   norm thresh           = %g\n",
				vl_sift_get_norm_thresh   (filt)) ;
			printf("vl_sift:   window size           = %g\n",
				vl_sift_get_window_size   (filt)) ;
			printf("vl_sift:   float descriptor      = %d\n",
				floatDescriptors) ;

			printf((nikeys >= 0) ?
				"vl_sift: will source frames? yes (%d read)\n" :
			"vl_sift: will source frames? no\n", nikeys) ;
			printf("vl_sift: will force orientations? %s\n",
				force_orientations ? "yes" : "no") ;
		}

		/* ...............................................................
		*                                             Process each octave
		* ............................................................ */
		i     = 0 ;
		first = 1 ;
		while (1) {
			int                   err ;
			VlSiftKeypoint const *keys  = 0 ;
			int                   nkeys = 0 ;

			if (verbose) {
				printf ("vl_sift: processing octave %d\n",
					vl_sift_get_octave_index (filt)) ;
			}

			/* Calculate the GSS for the next octave .................... */
			if (first) {
				err   = vl_sift_process_first_octave (filt, data) ;
				first = 0 ;
			} else {
				err   = vl_sift_process_next_octave  (filt) ;
			}

			if (err) break ;

			if (verbose > 1) {
				printf("vl_sift: GSS octave %d computed\n",
					vl_sift_get_octave_index (filt));
			}

			/* Run detector ............................................. */
			if (nikeys < 0) {
				vl_sift_detect (filt) ;

				keys  = vl_sift_get_keypoints  (filt) ;
				nkeys = vl_sift_get_nkeypoints (filt) ;
				i     = 0 ;

				if (verbose > 1) {
					printf ("vl_sift: detected %d (unoriented) keypoints\n", nkeys) ;
				}
			} else {
				nkeys = nikeys ;
			}

			/* For each keypoint ........................................ */
			for (; i < nkeys ; ++i) {
				double                angles [4] ;
				int                   nangles ;
				VlSiftKeypoint        ik ;
				VlSiftKeypoint const *k ;

				/* Obtain keypoint orientations ........................... */
				if (nikeys >= 0) {
					vl_sift_keypoint_init (filt, &ik,
						ikeys [4 * i + 1] - 1,
						ikeys [4 * i + 0] - 1,
						ikeys [4 * i + 2]) ;

					if (ik.o != vl_sift_get_octave_index (filt)) {
						break ;
					}

					k = &ik ;

					/* optionally compute orientations too */
					if (force_orientations) {
						nangles = vl_sift_calc_keypoint_orientations
							(filt, angles, k) ;
					} else {
						angles [0] = VL_PI / 2 - ikeys [4 * i + 3] ;
						nangles    = 1 ;
					}
				} else {
					k = keys + i ;
					nangles = vl_sift_calc_keypoint_orientations
						(filt, angles, k) ;
				}

				/* For each orientation ................................... */
				for (q = 0 ; q < nangles ; ++q) {
					vl_sift_pix  buf [128] ;
					vl_sift_pix rbuf [128] ;

					/* compute descriptor (if necessary) */
					vl_sift_calc_keypoint_descriptor (filt, buf, k, angles [q]) ;
					transpose_descriptor (rbuf, buf) ;

					/* make enough room for all these keypoints and more */
					if (reserved < (*nframes) + 1) {
						reserved += 2 * nkeys ;
						frames = (double*)realloc (frames, 4 * sizeof(double) * reserved) ;
						descr  = (vl_uint8*)realloc (descr,  128 * sizeof(vl_uint8) * reserved) ;
					}

					/* Save back with MATLAB conventions. Notice tha the input
					* image was the transpose of the actual image. */
					frames [4 * (*nframes) + 0] = k -> y ;
					frames [4 * (*nframes) + 1] = k -> x ;
					frames [4 * (*nframes) + 2] = k -> sigma ;
					frames [4 * (*nframes) + 3] = VL_PI / 2 - angles [q] ;

					for (j = 0 ; j < 128 ; ++j) {
						float x = 512.0F * rbuf [j] ;
						x = (x < 255.0F) ? x : 255.0F ;
						descr[128 * (*nframes) + j] = (vl_uint8)x ;
					}

					++ (*nframes) ;
				} /* next orientation */
			} /* next keypoint */
		} /* next octave */

		if (verbose) {
			printf ("vl_sift: found %d keypoints\n", (*nframes)) ;
		}

		if ((*nframes ) < 10000) {
		    // save variables:
		    memcpy(DATAframes, frames, 4 * (*nframes ) * sizeof(double));
		    memcpy(DATAdescr, descr, 128 * (*nframes ) * sizeof(vl_uint8));
		} else {
		    (*nframes ) = 0;
		}

		/* cleanup */
		vl_sift_delete (filt);

		free(frames);
	    free(descr);
	} /* end: do job */



	return;
}

void make_clustering(vl_uint8 *data, int N, int M, int K)
{
    int err = 0 ;

    vl_uint     *asgn = 0 ;
    vl_ikm_acc  *centers = 0 ;

    int method_type = VL_IKM_LLOYD ;
    int max_niters  = 200 ;
    int verb = 0 ;

    VlIKMFilt *ikmf =  vl_ikm_new (method_type);

    vl_ikm_set_verbosity  (ikmf, verb) ;
    vl_ikm_set_max_niters (ikmf, max_niters) ;
    vl_ikm_init_rand_data (ikmf, data, M, N, K) ;

    err = vl_ikm_train (ikmf, data, N) ;
    if (err) printf("ikmeans: possible overflow!") ;
/*
  {
    out[OUT_C] = mxCreateNumericMatrix (M, K, mxINT32_CLASS, mxREAL) ;
    centers    = mxGetData (out[OUT_C]) ;
    memcpy (centers, vl_ikm_get_centers (ikmf), sizeof(vl_ikm_acc) * M * K) ;
  }

  if (nout > 1) {
    int j ;
    out[OUT_I] = mxCreateNumericMatrix (1, N, mxUINT32_CLASS, mxREAL) ;
    asgn       = mxGetData (out[OUT_I]) ;

    vl_ikm_push (ikmf, asgn, data, N) ;

    for (j = 0 ; j < N ; ++j)
      ++ asgn [j] ;
  }

  vl_ikm_delete (ikmf) ;

  if (verb) {
    printf("ikmeans: done\n") ;
  }
*/
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

        for(int i = 0; i < Tnframes; i++) {
            circle(imgCv,
            cvPoint(TFrames[0+i*4], TFrames[1+i*4]), TFrames[2+i*4],
            cvScalar(255, 0, 0, 0),
            1, 8, 0);
        }

        /*
        cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
		cv::imshow("Display window", imgCv);
		cv::waitKey(0);
		*/

		NumImagesProcessed ++;

		if (NumImagesProcessed % 1000 == 0) cout << "processed " << NumImagesProcessed << endl;
	}

	cout << "Extracted " << ndescr << " descriptors" << endl;

	make_clustering(descr, ndescr, 128, 20);
}
