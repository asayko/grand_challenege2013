function [image3] = imgcat(image1, image2)
iheight = max(size(image1,1),size(image2,1));
if size(image1,1) < iheight
    image1(end+1:iheight,:,:) = 0;
end
if size(image2,1) < iheight
    image2(end+1:iheight,:,:) = 0;
end
image3 = cat(2,image1,image2);

end

