inpath='d:/toxas/images_jpeg_renamed/'
outpath='feats/'
tmp ='tmp/'

files = [dir([inpath '*.jpeg'])];
%%
parfor i=1:matlabpool('size')
        vl_setup;
end

vl_setup
%%
load('vocabs/vocab_l216384.mat');
kdtree = vl_kdtreebuild(vocab) ;

%%
z =  1:16384;
v = int32(vocab);
parfor i=1:numel(files)
    i
    histFile = [tmp files(i).name '_hist.mat'];
    if ~exist(histFile, 'file')
        try
            featureFile = [outpath files(i).name '.mat'];
            [f,d] = loadd(featureFile);
            %dd = single(d);
            %[idxs, distance] = vl_kdtreequery(kdtree, vocab, dd);
            idxs =  vl_ikmeanspush(d,v);
            hist = histc(idxs, z);
    
            fakeSave(histFile, hist);
        catch exc
           %do nothing
        end
    end;
end
