inpath='./'
outpath=''

files = [dir([inpath 'vocab*mat'])];

for i=1:numel(files)
   outname = [outpath files(i).name '.dat'];
   load([inpath files(i).name]);
   dim = size(vocab, 1);
   sz = size(vocab, 2);
   file = fopen(outname, 'w');
   fwrite(file, 1, 'int32');
   fwrite(file, sz, 'int32');
   fwrite(file, dim, 'int32');
   fwrite(file, 0, 'int32');
   fwrite(file, vocab, 'float');
   fclose(file);
end
