%% ================== 局部函数：特征提取 ==================
function feat = extractFlowerFeature(imgPath, net, inputSize, featureLayer, colorWeight, textureWeight)

    img = imread(imgPath);

    if size(img, 3) == 1
        img = cat(3, img, img, img);
    end

    img = im2uint8(img);
    img_raw = img;

    %% CNN深度特征
    img_cnn = imresize(img, inputSize(1:2));

    cnnFeat = activations(net, img_cnn, featureLayer, ...
        'OutputAs', 'rows');

    cnnFeat = normalize(cnnFeat, 2, 'norm');

    %% HSV颜色直方图特征
    img_color = imresize(img_raw, [224 224]);
    hsvImg = rgb2hsv(img_color);

    hHist = imhist(hsvImg(:,:,1), 32);
    sHist = imhist(hsvImg(:,:,2), 32);
    vHist = imhist(hsvImg(:,:,3), 32);

    colorFeat = [hHist; sHist; vHist]';
    colorFeat = colorFeat / (sum(colorFeat) + eps);

    %% LBP纹理特征
    img_texture = imresize(img_raw, [224 224]);
    grayImg = rgb2gray(img_texture);

    lbpFeat = extractLBPFeatures(grayImg, ...
        'NumNeighbors', 8, ...
        'Radius', 1, ...
        'Upright', false);

    lbpFeat = lbpFeat / (sum(lbpFeat) + eps);

    %% 特征融合
    feat = [cnnFeat, colorWeight * colorFeat, textureWeight * lbpFeat];
end