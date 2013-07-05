inpath='d:/toxas/images_jpeg_renamed_dev/'
outpath='feats_dev/'
tmp ='hists_idxs_dev/'

files = [dir([inpath '*.jpeg'])];
%save('files');

fout = fopen('hist_32768_l2.txt', 'w');
z = 1:32768;
for i=1:numel(files)
    i
    histFile = [tmp files(i).name '_idxs_32768.mat'];
    fprintf(fout, '%s\t', files(i).name);
    try
        load(histFile);
        hist = histc(hist, z);
        idx = find(hist > 0);
        for k=1:numel(idx)
         fprintf(fout, '%d,%d;',idx(k), hist(idx(k))); 
        end
        fprintf(fout,'\n');
    except
    end
end
fclose(fout);