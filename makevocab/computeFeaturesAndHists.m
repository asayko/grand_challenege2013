insrc='~/semidups-20130625-171944/results/semidups-000.workd/thumbnails-prod/'
%featspath='~/semidups-20130625-171944/results/semidups-000.workd/thumbnails-beta/'
%histspath=featspath;
dups='~/semidups-20130625-171944/dups-prod-32768/';
mkdir(dups);
vocabpath = 'vocabs/vocab_l232768.mat';
load(vocabpath);

for i=0:100
    sprintf('folder:%d',i)
    inpath = [insrc int2str(i) '/'];
    featspath = [inpath 'tmp/'];
    histspath = featspath;
    mkdir(featspath);
    computeFeaturesImpl(inpath, featspath);
    calcHistsImpl(inpath, featspath, vocab, histspath);
    dupF = [dups int2str(i)];
    mkdir(dupF);
    genGroups(histspath, [dupF '/dups.txt'], 100, 0.3, size(vocab,2));
end

dups='~/semidups-20130625-171944/dups-prod-16384/';
mkdir(dups);
vocabpath = 'vocabs/vocab_l216384.mat';
load(vocabpath);


for i=0:100
    sprintf('folder:%d',i)
    inpath = [insrc int2str(i) '/'];
    featspath = [inpath 'tmp/'];
    histspath = featspath;
    mkdir(featspath);
    computeFeaturesImpl(inpath, featspath);
    calcHistsImpl(inpath, featspath, vocab, histspath);
    dupF = [dups int2str(i)];
    mkdir(dupF);
    genGroups(histspath, [dupF '/dups.txt'], 100, 0.3, size(vocab,2));
end