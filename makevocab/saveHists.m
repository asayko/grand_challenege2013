inpath='d:/toxas/images_jpeg_renamed_dev/'
outpath='feats_dev/'
tmp ='tmp_dev/'

files = [dir([inpath '*.jpeg'])];
%save('files');

fout = fopen('hist_16384_l2.txt', 'w');
for i=1:numel(files)
    i
    histFile = [tmp files(i).name '_hist.mat'];
    fprintf(fout, '%s\t', files(i).name);
    try
        load(histFile);
        idx = find(hist > 0);
        for k=1:numel(idx)
         fprintf(fout, '%d,%d;',idx(k), hist(idx(k))); 
        end
        fprintf(fout,'\n');
    except
    end
end
fclose(fout);