inpath='~/grand_challenege2013/images_jpeg_renamed_dev/'
outpath='feats_dev/'
tmp ='hists_idxs_dev/'

files = [dir([inpath '*.jpeg'])];
%save('files');

fout = fopen('hist_32768_l2.txt', 'w');

foutF = fopen('hist_32768_l2_f.txt', 'w');
z = 1:32768;
for i=1:numel(files)
    i
    histFile = [tmp files(i).name '_idxs_32768.mat'];
    if exist(histFile, 'file')
        fprintf(foutF, '%s\t', files(i).name);
        fFile = [tmp files(i).name '_f_32768.mat'];
        load(fFile);
        f = hist;
        load(histFile);

        [~,idxs] = sort(hist);
        
        for k=1:numel(idxs);
         i = idxs(k);
         fprintf(foutF, '%d,%f,%f,%f;',hist(i), f(1,i), f(2,i), f(3,i)); 
        end
        %%
        hist = histc(hist, z);
        idx = find(hist > 0);
        fprintf(fout, '%s\t', files(i).name);


        for k=1:numel(idx)
         fprintf(fout, '%d,%d;',idx(k), hist(idx(k)));
        end
        fprintf(fout,'\n');
        fprintf(foutF, 'n');
    end
end
fclose(fout);
fclose(foutF);