run('/Users/slesarev/vlfeat-0.9.14/toolbox/vl_setup.m')

Ia = iresize(imread('imgs/i2.jpeg'));
Ib = iresize(imread('imgs/x4.jpeg'));

% distimages('imgs/g1.jpeg', 'imgs/g2.jpeg')
% distimages('imgs/g2.jpeg', 'imgs/i3.jpeg')
% distimages('imgs/g1.jpeg', 'imgs/i4.jpeg')
% distimages('imgs/i4.jpeg', 'imgs/i5.jpeg')
% 
% distimages('imgs/i1.jpeg', 'imgs/i2.jpeg')
% distimages('imgs/i1.jpeg', 'imgs/i3.jpeg')
% distimages('imgs/i3.jpeg', 'imgs/i4.jpeg')
% distimages('imgs/i4.jpeg', 'imgs/i5.jpeg')

im1 = single(rgb2gray(Ia));
im2 = single(rgb2gray(Ib));

load('vocabs/vocab_l216384.mat');

kdtree = vl_kdtreebuild(vocab) ;
[f1, idx1] = hist(im1, kdtree, vocab);
[f2, idx2] = hist(im2, kdtree, vocab);
[l r]=find(bsxfun(@eq,idx1,idx2')');

dist = histDist(idx1, idx2);
dist

figure(1) ; clf ;
imagesc(cat(2, Ia, Ib)) ;

xa = f1(1,l) ;
xb = f2(1,r) + size(Ib,2) ;
ya = f1(2,l) ;
yb = f2(2,r) ;

hold on ;
h = line([xa ; xb], [ya ; yb]) ;
set(h,'linewidth', 1, 'color', 'b') ;

vl_plotframe(f1(:,l)) ;
f2(1,:) = f2(1,:) + size(Ia,2) ;
vl_plotframe(f2(:,r)) ;
axis image off ;