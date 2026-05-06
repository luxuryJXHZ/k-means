clc; clear; close all;

%% 1. 图像路径
imgDir = fullfile(pwd, 'fl');
outDir = fullfile(pwd, 'kmeans_result1');

if exist(outDir, 'dir')
    rmdir(outDir, 's');
end
mkdir(outDir);

files = [ ...
    dir(fullfile(imgDir, '*.jpg')); ...
    dir(fullfile(imgDir, '*.png')); ...
    dir(fullfile(imgDir, '*.jpeg')) ...
];

numImages = numel(files);
fprintf('共读取到 %d 张图像。\n', numImages);

%% 2.网络
        net = mobilenetv2;
        inputSize = net.Layers(1).InputSize;
        featureLayer = 'Logits';
        featDim =1000 + 96 + 59;

 
features = [];

%% 3. 特征提取：CNN深度特征 + HSV颜色特征 + LBP纹理特征5
colorWeight = 4;
textureWeight = 1;

for i = 1:numImages
    imgPath = fullfile(files(i).folder, files(i).name);
    img = imread(imgPath);

    % 灰度图转RGB
    if size(img, 3) == 1
        img = cat(3, img, img, img);
    end

    % 转为uint8，防止部分图像格式异常
    img = im2uint8(img);

    % 保留一份原图用于颜色/纹理特征
    img_raw = img;

    % CNN输入图像必须resize到网络输入大小
    img_cnn = imresize(img, inputSize(1:2));

    %% ===== 1. CNN深度特征 =====
    cnnFeat = activations(net, img_cnn, featureLayer, ...
        'OutputAs', 'rows');

    cnnFeat = normalize(cnnFeat, 2, 'norm');

    %% ===== 2. HSV颜色直方图特征 =====
    img_color = imresize(img_raw, [224 224]);
    hsvImg = rgb2hsv(img_color);

    hHist = imhist(hsvImg(:,:,1), 32);
    sHist = imhist(hsvImg(:,:,2), 32);
    vHist = imhist(hsvImg(:,:,3), 32);

    colorFeat = [hHist; sHist; vHist]';
    colorFeat = colorFeat / (sum(colorFeat) + eps);

    %% ===== 3. LBP纹理特征 =====
    img_texture = imresize(img_raw, [224 224]);
    grayImg = rgb2gray(img_texture);

    lbpFeat = extractLBPFeatures(grayImg, ...
        'NumNeighbors', 8, ...
        'Radius', 1, ...
        'Upright', false);

    lbpFeat = lbpFeat / (sum(lbpFeat) + eps);

    %% ===== 4. 特征融合 =====
    feat = [cnnFeat, colorWeight * colorFeat, textureWeight * lbpFeat];

   if i == 1
    features = zeros(numImages, length(feat));
    end

features(i, :) = feat;

    fprintf('正在处理第 %d / %d 张：%s\n', ...
        i, numImages, files(i).name);
end

%% 4. 提取真实标签
trueLabels = strings(numImages, 1);
classNames = ["daisy", "dandelion", "roses", "sunflowers", "tulips"];

for i = 1:numImages
    name = lower(files(i).name);

    for j = 1:length(classNames)
        if contains(name, classNames(j))
            trueLabels(i) = classNames(j);
            break;
        end
    end
end

%% 5. 原始特征先归一化
features = normalize(features, 2, 'norm');

%% 6. 自动搜索最佳PCA维度（k固定为5）
[~, score] = pca(features);
k_cluster = 5;

maxDim = size(score, 2);
dimList = 5:5:800;
dimList = dimList(dimList <= maxDim);   % 防止越界

bestAcc = 0;
bestErrRate = 1;
bestDim = 0;
bestIdx = [];
bestFeatures = [];
bestSC = -inf;

accList = zeros(size(dimList));
errList = zeros(size(dimList));
scList = zeros(size(dimList));

fprintf('\n========== 开始搜索最佳PCA维度 ==========\n');

for d = 1:length(dimList)

    dim = dimList(d);

    features_pca = score(:, 1:dim);

    % L2归一化（关键）
    features_pca = normalize(features_pca, 2, 'norm');
    initC = farthestInit(features_pca, k_cluster);

    [idx, ~, ~] = kmeans(features_pca, k_cluster, ...
    'Start', initC, ...
    'Distance', 'cosine', ...
    'MaxIter', 2000, ...
    'Display', 'off');

    % 轮廓系数
    sc = mean(silhouette(features_pca, idx));

    % 正确率
    [acc, errRate] = calcClusterAccuracy(idx, trueLabels, classNames);

    accList(d) = acc;
    errList(d) = errRate;
    scList(d) = sc;

    fprintf('PCA维度 = %3d | 正确率 = %.2f%% | 错误率 = %.2f%% | SC = %.4f\n', ...
        dim, acc * 100, errRate * 100, sc);

    % 更新最优结果
    if acc > bestAcc
        bestAcc = acc;
        bestErrRate = errRate;
        bestDim = dim;
        bestIdx = idx;
        bestFeatures = features_pca;
        bestSC = sc;
    end
end

% 赋值最佳结果
idx = bestIdx;
features = bestFeatures;

fprintf('\n========== 最佳结果 ==========\n');
fprintf('最佳PCA维度：%d\n', bestDim);
fprintf('最佳正确率：%.2f%%\n', bestAcc * 100);
fprintf('最佳错误率：%.2f%%\n', bestErrRate * 100);
fprintf('最佳SC轮廓系数：%.4f\n', bestSC);
%% 7. 保存最佳聚类结果
cluster_num = max(idx);
clusterDirs = cell(cluster_num, 1);

for c = 1:cluster_num
    clusterDirs{c} = fullfile(outDir, ['cluster_', num2str(c)]);

    if ~exist(clusterDirs{c}, 'dir')
        mkdir(clusterDirs{c});
    end
end

for i = 1:numImages
    srcPath = fullfile(files(i).folder, files(i).name);
    cluster_id = idx(i);
    dstPath = fullfile(clusterDirs{cluster_id}, files(i).name);

    copyfile(srcPath, dstPath);
end

fprintf('\n分类结果已保存到：%s\n', outDir);

%% 8. 输出每个cluster统计结果
for c = 1:cluster_num
    count_cluster_general(clusterDirs{c});
end

fprintf('\nK-means聚类完成！\n');

for c = 1:cluster_num
    fprintf('cluster_%d 图像数量：%d\n', c, sum(idx == c));
end



