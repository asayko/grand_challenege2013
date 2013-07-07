function [ out ] = iresize( in)
           ratio = 320/max(size(in));
           out = in;
           if ratio < 1
               out = imresize(im, ratio);
           end


end

