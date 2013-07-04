function [dist] = histRawDist(hist1, hist2)
    den = sum(hist1 + hist2);
    num = sum(hist2.*(hist1 > 0) + hist1.*(hist2 > 0));
    %den
    %num
    dist = num / den;
end

