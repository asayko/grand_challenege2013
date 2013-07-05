vocabpath = 'vocabs/vocab_l232768.mat';
load(vocabpath);

inpath='d:/toxas/images_jpeg_renamed/';
featspath = 'feats/';
histspath = 'hists_idxs/';
mkdir(featspath);
%computeFeaturesImpl(inpath, featspath);
calcHistsImpl(inpath, featspath, vocab, histspath);
