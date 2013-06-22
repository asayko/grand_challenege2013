#include "my_sift.h"

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

void VLSIFT(cv::Mat* image, vl_uint8* DATAdescr, double* DATAframes, int* nframes, int verbose ){
	//Take IplImage -> convert to SINGLE (float):
	float* frame = (float*)malloc(image->rows * image->cols * sizeof(float));
    uchar* Ldata = (uchar*)image->data;

	for(int i = 0; i < image->rows; i++)
		for(int j = 0; j < image->cols; j++)
			frame[j*image->rows + i] = (float)Ldata[i*image->step + j];

	// VL SIFT computation:
    vl_sift_pix const *data = (vl_sift_pix*)frame;
    int M = image->rows, N = image->cols;

	int                O     =   -1 ; //Octaves
	int                S     =   3 ;  //Levels
	int                o_min =   0 ;

	double             edge_thresh = -1 ;
    double             peak_thresh = 2.0 ;
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

    free(frame);
	return;
}
