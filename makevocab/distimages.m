function [dist, n1, n2] = distimages(i1, i2)

Ia = iresize(imread(i1));

Ib = iresize(imread(i2));
im1 = single(rgb2gray(Ia));
im2 = single(rgb2gray(Ib));

%load('vocabs/vocab_l216384.mat');
load('vocabs/vocab_l232768.mat');

kdtree = vl_kdtreebuild(vocab);
[f1, idx1] = histFromImage(im1, kdtree, vocab);
[f2, idx2] = histFromImage(im2, kdtree, vocab);
[l r] = find(bsxfun(@eq,idx1,idx2')');

dist = histDist(idx1, idx2, size(vocab,2));
n1 = numel(idx1);
n2 = numel(idx2);
end

