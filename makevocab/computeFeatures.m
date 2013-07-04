function ComputeFeatures
inpath='d:/toxas/images_jpeg_renamed_dev/'
outpath='feats_dev/'
mkdir(outpath)
computeFeaturesImpl(inpath, outpath);
