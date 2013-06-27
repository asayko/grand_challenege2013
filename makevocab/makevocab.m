vl_setup;
addpath('D:\vilemp\db-helper\');
% 
% folder = 'feats/';
% files=dir([folder '*mat']);
% 
% descrs = {};
% descrsL = {};
% 
% for i=1:numel(files)
%     if  mod(i, 10000) == 0
%         i
%         descrsL{1 + i / 10000} = vl_colsubset(cat(2, descrs{:}), 500000);
%         descrs = {};
%     end
%     f = load([folder files(i).name]);
%     descrs{1 + mod(i,10000)} = f.d;
% end
% 
% %%
%load descrsL
% x = [];
% for i=1:10:numel(descrsL)
%     x = [x vl_colsubset(cat(2, descrsL{i:i+10}), 2000000)];
% end
%%

%descrsL2 = vl_colsubset(x, 45000000);
%x = single() ;
%save('descrsL2.mat', 'descrsL2')
%all_data = fvecs_read('sift1K/sift_base.fvecs');
%run('D:\toxas\vlfeat-0.9.13\toolbox\vl_setup')
baseVocabSize = [2^10 2^14 2^15 2^18 2^20];
for i = 1:numel(baseVocabSize)
    vocabSize = baseVocabSize(i)
    load descrsL2.mat
    data = uint8(vl_colsubset(x, vocabSize*40));
    clear x
    vocab = vl_hikmeans(data,vocabSize, vocabSize);
    save(['vocab_l2_hikm' num2str(vocabSize) '.mat'], 'vocab');
end
    
    