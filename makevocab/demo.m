run('/Users/slesarev/vlfeat-0.9.14/toolbox/vl_setup.m')

Ia = iresize(imread('~/semidups-20130625-171944/results/semidups-000.workd/thumbnails-prod/45/1.jpg'));
Ib = iresize(imread('~/semidups-20130625-171944/results/semidups-000.workd/thumbnails-prod/45/18.jpg'));


%Ia = iresize(imread('imgs/f1.jpeg'));
%Ib = iresize(imread('imgs/f2.jpeg'));

%distimages('imgs/f1.jpeg', 'imgs/f2.jpeg')
%distimages('imgs/f2.jpeg', 'imgs/f3.jpeg')
%distimages('imgs/g1.jpeg', 'imgs/i4.jpeg')
% distimages('imgs/i4.jpeg', 'imgs/i5.jpeg')
% 
%%
% fout = fopen('tmp_raw_2.txt', 'w');
% for i=1:5
%     for j=1:5
%         [score, n1, n2] = distimages(['imgs/b' int2str(i) '.jpeg'], ['imgs/b' int2str(j) '.jpeg']);
%         fprintf(fout, '%d %d %f %d %d\n', i, j, score, n1, n2);
%     end
% end
% fclose(fout);
%%

im1 = single(rgb2gray(Ia));
im2 = single(rgb2gray(Ib));

load('vocabs/vocab_l216384.mat');
load('vocabs/vocab_l232768.mat');

kdtree = vl_kdtreebuild(vocab) ;
[f1, idx1] = histFromImage(im1, kdtree, vocab);
[f2, idx2] = histFromImage(im2, kdtree, vocab);
[l r]=find(bsxfun(@eq,idx1,idx2')');

dist = histDist(idx1, idx2, size(vocab,2));
dist
sum(idx1>0)
sum(idx2>0)
sum((histc(idx1,1:size(vocab,2))>0).*(histc(idx2,1:size(vocab,2))>0))

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