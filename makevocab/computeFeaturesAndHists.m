insrc='~/semidups-20130625-171944/results/semidups-000.workd/thumbnails-prod/'
%featspath='~/semidups-20130625-171944/results/semidups-000.workd/thumbnails-beta/'
%histspath=featspath;
dups='~/semidups-20130625-171944/dups-prod/';
mkdir(dups);

for i=0:100
    sprintf('folder:%d',i)
    inpath = [insrc int2str(i) '/'];
    featspath = [inpath 'tmp/'];
    histspath = featspath;
    mkdir(featspath);
    computeFeaturesImpl(inpath, featspath);
    calcHistsImpl(inpath, featspath, histspath);
    vocab=load('vocabs/vocab_l216384.mat');
    dupF = [dups int2str(i)];
    mkdir(dupF);
    genGroups(histspath, [dupF '/dups.txt'], 20, 0.2);
end