function [ dist] = distimages(i1, i2 )

Ia = iresize(imread(i1));

Ib = iresize(imread(i2));
im1 = single(rgb2gray(Ia));
im2 = single(rgb2gray(Ib));

load('vocab_l216384.mat');

kdtree = vl_kdtreebuild(vocab);
[f1, idx1] = hist(im1, kdtree, vocab);
[f2, idx2] = hist(im2, kdtree, vocab);
[l r]=find(bsxfun(@eq,idx1,idx2')');

dist = histDist(idx1, idx2);
end

