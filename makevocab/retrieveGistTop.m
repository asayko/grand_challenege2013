run('/Users/slesarev/vlfeat-0.9.14/toolbox/vl_setup.m')
inpath='~/grand_challenege2013/images_jpeg_renamed_dev/'
outpath='feats_dev/'
tmp ='hists_idxs_dev/'

files = [dir([inpath '*.jpeg'])];
%save('files');
%gists = {}
clear param
param.imageSize = [256 256]; % it works also with non-square images
param.orientationsPerScale = [8 8 8 8];
param.numberBlocks = 4;
param.fc_prefilt = 4;

for i=1:150
    i
    im = imread([inpath files(i).name]);
    [gist, ~] = LMgist(im, '', param);
    gists{i} = gist;
end

%
q = 'aW1nMjA0MjQ=.jpeg'

qim = imread([inpath q]);
[gist, ~] = LMgist(qim, '', param);

scores = zeros(150, 1);
for i=1:150
    scores(i) = pdist2(gist,gists{i},'euclidean');
end

[~, scoresIdx] = sort(scores, 'ascend');

figure, imshow(qim);
w = waitforbuttonpress;
%%
for i=1:10
    resIdx = scoresIdx(i);
    scores(resIdx)
    files(resIdx).name
    im = imread([inpath files(resIdx).name]);
    figure, imshow(im);
    w = waitforbuttonpress;
end
close all




