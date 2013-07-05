function calcHistsImpl(inpath, featpath, vocab,  tmp)
files = [dir([inpath '*.jpg']); dir([inpath '*.png']); dir([inpath '*.jpeg'])];

%%
parfor i=1:matlabpool('size')
        vl_setup;
end

vl_setup
%%
kdtree = vl_kdtreebuild(vocab) ;

%%
z =  1:size(vocab,2);
v = int32(vocab);
vsize = size(vocab, 2);
sprintf('compute hists')
mkdir(tmp);
parfor i=1:numel(files)
    histFile = [tmp files(i).name '_hist_' int2str(vsize) '.mat'];
    idxsFile = [tmp files(i).name '_idxs_' int2str(vsize) '.mat'];
    fFile = [tmp files(i).name '_f_' int2str(vsize) '.mat'];
    i
    if ~exist(idxsFile, 'file')
        try
            featureFile = [featpath files(i).name '.mat'];
            [f,d] = loadd(featureFile);
            %dd = single(d);
            %[idxs, distance] = vl_kdtreequery(kdtree, vocab, dd);
            idxs =  vl_ikmeanspush(d,v);
            fakeSave(idxsFile, idxs);
            fakeSave(fFile, f);
            %hist = histc(idxs, z);
    
            %fakeSave(histFile, hist);
        catch exc
           %do nothing
        end
    end;
end

end

