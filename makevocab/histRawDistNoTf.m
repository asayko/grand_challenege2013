function [score] = histRawDistNoTf(hist1, hist2)
    den = sum((hist1>0) + (hist2>0));
    num = sum(2 * (hist2>0).*(hist1 > 0));
    %den
    %num
    score = num / den;
end

