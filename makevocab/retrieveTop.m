run('/Users/slesarev/vlfeat-0.9.14/toolbox/vl_setup.m')
inpath='~/grand_challenege2013/images_jpeg_renamed_dev/'
outpath='feats_dev/'
hists_path ='hists_idxs_dev/'

%%load index
%%hists or raw hists?
%%sparse hists?
files = [dir([inpath '*.jpeg'])];
%save('files');
z = 1:32768;
%hists = uint16(zeros(32768,20000));
hists = {}
histCnt = numel(files);
for i=1:histCnt
    i
    histFile = [hists_path files(i).name '_idxs_32768.mat'];
    if exist(histFile, 'file')
        load(histFile);
        hists{i} = hist;
        %hists(:,i) = histc(hist, z);
    end
end

%%
%%query

q = 'aW1nMjA4MDE=.jpeg'
load('vocabs/vocab_l232768.mat');
kdtree = vl_kdtreebuild(vocab) ;
im = imread([inpath q]);
im = rgb2gray(im);
ratio = 320/max(size(im));
if ratio < 1
    im = imresize(im, ratio);
end
sz = size(im);
%tic

[f1, idx1] = histFromImage(im, kdtree, vocab);

qHist = histc(idx1, 1:32768);
% qHist = uint16(qHist);
% qHist = qHist';
% %%
% tic
% scores = vl_alldist2(qHist, hists);
% toc
% 
% %%
% sums = sum(hists,1);
%scores = double(scores) ./ (1 + double(sums));

tic
scores = zeros(numel(hists), 1);
z = 1:32768;
for i=1:numel(hists)
    if size(hists{i}, 1) > 0
        histRaw = histc(hists{i}, z);
        scores(i) = histRawDist(qHist, histRaw);
    end
end
toc


[~, scoresIdx] = sort(scores, 'descend');

%%
%%draw
for i=1:5
    resIdx = scoresIdx(i);
    scores(resIdx)
    files(resIdx).name
    im = imread([inpath files(resIdx).name]);
    figure, imshow(im);
    w = waitforbuttonpress;
end



