function genGroups(histspath, out, topN, thr, vocabSize);

parfor i=1:matlabpool('size')
        vl_setup;
end

fout = fopen(out, 'w');
fout_debug = fopen([out '.debug'], 'w');

hists = {}
for i=0:topN
   histname = [histspath int2str(i) '.jpg_hist_' int2str(vocabSize) '.mat'];

   try
   tmp = load(histname);
   hists{i+1} = tmp(1).hist;
   catch
   hists{i+1} = zeros(1,vocabSize);
   end

end

curGroup = 1;
groups = {};
isDup = 0;
for i=0:topN
    isDup = 0;
   for j=0:i-1
       score = histRawDist(hists{i+1}, hists{j+1});
       fprintf(fout_debug, '%d.jpg,%d.jpg,%1.3f,%d\n', i, j, score, sum(hists{j+1}>0));
  
       if (score > thr) && (sum(hists{j+1} > 0) > 20) && (sum(hists{i+1} > 0) > 20)
           isDup = 1;
           fprintf(fout, '1\t%d\n', groups{j+1});
           groups{i+1} = groups{j + 1};
           break;
       end
          
   end
   if isDup == 0
    fprintf(fout, '0\t%d\n', curGroup);
    groups{i+1} = curGroup;
    curGroup = curGroup + 1;
   end
end
fclose(fout);
fclose(fout_debug);
end

