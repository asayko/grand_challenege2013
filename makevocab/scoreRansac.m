function [dist] = scoreRansac(i1, i2 )

Ia = iresize(imread(i1));
Ib = iresize(imread(i2));

im1 = single(rgb2gray(Ia));
im2 = single(rgb2gray(Ib));

load('vocabs/vocab_l232768.mat');

kdtree = vl_kdtreebuild(vocab);
[f1, idx1] = histFromImage(im1, kdtree, vocab);
[f2, idx2] = histFromImage(im2, kdtree, vocab);

hdist = histDist(idx1, idx2, size(vocab,2));

hestgeotform = vision.GeometricTransformEstimator;

if hdist < 0.05
    dist = 0;
else
   [l r]=find(bsxfun(@eq,idx1,idx2')');
   [tform, inliers] = step(hestgeotform,f1(1:2,l)', f2(1:2,r)');
   dist = sum(inliers);
end

