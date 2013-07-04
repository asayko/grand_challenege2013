function calcHistsImpl(inpath, featpath, tmp)
files = [dir([inpath '*.jpg']); dir([inpath '*.png']); dir([inpath '*.jpeg'])];

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
sprintf('compute features');
parfor i=1:numel(files)
    histFile = [tmp files(i).name '_hist.mat'];
    if ~exist(histFile, 'file')
        try
            featureFile = [featpath files(i).name '.mat'];
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

end

