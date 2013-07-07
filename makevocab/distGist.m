function [dist] = distGist(ifile1, ifile2)
addpath('gistdescriptor');
clear param
param.imageSize = [256 256]; % it works also with non-square images
param.orientationsPerScale = [8 8 8 8];
param.numberBlocks = 4;
param.fc_prefilt = 4;
im1 = imread(ifile1);
im2 = imread(ifile2);
[gist1, ~] = LMgist(im1, '', param);
[gist2, ~] = LMgist(im2, '', param);
dist = pdist2(gist1,gist2,'euclidean');
end


