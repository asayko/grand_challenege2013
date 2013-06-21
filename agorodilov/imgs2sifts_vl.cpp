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

// g++ imgs2sifts_vl.cpp -I /Users/agorodilov/work/msr_image/vlfeat-0.9.16/ `pkg-config opencv --cflags --libs` /Users/agorodilov/work/msr_image/vlfeat-0.9.16/bin/maci64/libvl.dylib  -o imgs2sifts_vl

typedef struct
{
	int k1 ;
	int k2 ;
	double score ;
} Pair ;


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


static int
	korder (void const* a, void const* b) {
		double x = ((double*) a) [2] - ((double*) b) [2] ;
		if (x < 0) return -1 ;
		if (x > 0) return +1 ;
		return 0 ;
}


vl_bool
	check_sorted (double const * keys, vl_size nkeys)
{
	vl_uindex k ;
	for (k = 0 ; k + 1 < nkeys ; ++ k) {
		if (korder(keys, keys + 4) > 0) {
			return VL_FALSE ;
		}
		keys += 4 ;
	}
	return VL_TRUE ;
}


void VLSIFT(IplImage* image, vl_uint8* DATAdescr, double* DATAframes, int* nframes){
	//Take IplImage -> convert to SINGLE (float):
	float* frame = (float*)malloc(image->height*image->width*sizeof(float));
	uchar* Ldata      = (uchar *)image->imageData;
	for(int i=0;i<image->height;i++)
		for(int j=0;j<image->width;j++)
			frame[j*image->height+i*image->nChannels] = (float)Ldata[i*image->widthStep+j*image->nChannels];
	/*
	FILE *fpp = fopen("c:\\Picture.txt", "w");
		for(int p=0;p<image->height*image->width; p++){
			fprintf(fpp, "%f\n",frame[p] );
		}
		fclose(fpp);
		*/

	// VL SIFT computation:
	vl_sift_pix const *data ;
	int                M, N ;
	data = (vl_sift_pix*)frame;
	M = image->height;
	N = image->width;

	int                verbose = 1 ;
	int                O     =   -1 ; //Octaves
	int                S     =   3 ; //Levels
	int                o_min =   0 ;

	double             edge_thresh = -1 ;
	double             peak_thresh =  -1 ;
	double             norm_thresh = -1 ;
	double             magnif      = -1 ;
	double             window_size = -1 ;

	//mxArray           *ikeys_array = 0 ; //?
	double            *ikeys = 0 ; //?
	int                nikeys = -1 ; //?
	vl_bool            force_orientations = 0 ;
	vl_bool            floatDescriptors = 0 ;

	/* -----------------------------------------------------------------
	*                                                            Do job
	* -------------------------------------------------------------- */
	{
		VlSiftFilt        *filt ;
		vl_bool            first ;
		double            *frames = 0 ;
		vl_uint8              *descr  = 0 ;
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
		// save variables:
		memcpy(DATAframes, frames, 4 * (*nframes ) * sizeof(double));
		memcpy(DATAdescr, descr, 128 * (*nframes ) * sizeof(vl_uint8));

		/*
		FILE *fpd = fopen("c:\\Descr.txt", "w");
		for(int p=0;p<(*nframes)*128; p++){
			fprintf(fpd, "%f\n",(double)descr[p] );
		}
		fclose(fpd);

		FILE *fpf = fopen("c:\\Frames.txt", "w");
		for(int p=0;p<(*nframes)*4; p++){
			fprintf(fpf, "%f\n",frames[p] );
		}
		fclose(fpf);
		*/

		/* cleanup */
		vl_sift_delete (filt) ;
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
	IplImage* Timage = cvLoadImage("input.jpg", 0);
	double*            TFrames = (double*)calloc ( 4 * 10000, sizeof(double) ) ;
	vl_uint8*          TDescr  = (vl_uint8*)calloc ( 128 * 10000, sizeof(vl_uint8) ) ;
	int                Tnframes = 0;
	VLSIFT(Timage, TDescr, TFrames, &Tnframes);
	TFrames = (double*)realloc (TFrames, 4 * sizeof(double) * Tnframes) ; // = Y X Scale Angle
	TDescr  = (vl_uint8*)realloc (TDescr,  128 * sizeof(vl_uint8) * Tnframes) ;

	for(int i=0;i<Tnframes;i++){
		cvCircle(Timage,
		cvPoint(TFrames[0+i*4], TFrames[1+i*4]), TFrames[2+i*4],
		cvScalar(255, 0, 0, 0),
		1, 8, 0);
	}

	make_clustering(TDescr, Tnframes, 128, 20);

	cvShowImage("FrameT", Timage);
	cvWaitKey(0);
}
