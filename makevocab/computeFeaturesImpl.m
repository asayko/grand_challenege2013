function computeFeaturesImpl(inpath, outpath)

files = [dir([inpath '*.jpg']); dir([inpath '*.png']); dir([inpath '*.jpeg'])];

parfor i=1:matlabpool('size')
        vl_setup;
end

parfor i=1:numel(files)
   outname = [outpath files(i).name '.mat'];
   if ~exist(outname, 'file')
       try
           im = imread([inpath files(i).name]);
           im = rgb2gray(im);
           ratio = 320/max(size(im));
           if ratio < 1
               im = imresize(im, ratio);
           end
           sz = size(im);
                %tic
           f = vl_sift(single(im));
           f(end,:) = 0;
           [f d] = vl_sift(single(im), 'Frames', f);
                %toc
           save_(outname, f, d, sz);
        catch exc
                %do nothing
       end
   end
end
end

function save_(outname, f, d, sz)
    save(outname, 'f','d','sz');
end

