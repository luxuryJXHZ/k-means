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

%% 2. 加载 MobileNetV2
net = mobilenetv2;
inputSize = net.Layers(1).InputSize;

% 建议特征层
featureLayer = 'global_average_pooling2d_1';
features = zeros(numImages, 1280);

%% 3. 逐张提取特征
for i = 1:numImages
    imgPath = fullfile(files(i).folder, files(i).name);
    img = imread(imgPath);

    if size(img, 3) == 1
        img = cat(3, img, img, img);
    end

    img = imresize(img, inputSize(1:2));

    feat = activations(net, img, featureLayer, ...
        'OutputAs', 'rows');

    
    features(i, :) = feat;

    fprintf('正在处理第 %d / %d 张：%s\n', ...
        i, numImages, files(i).name);
end

%% 4. 提取真实类别标签
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

%% 5. PCA降维
[~, score, latent] = pca(features);

%% 6. 自动搜索最佳PCA维度
k_cluster = 5;

dimList = 5:5:100;

bestAcc = 0;
bestErrRate = 1;
bestDim = 0;
bestIdx = [];
bestFeatures = [];
bestSC = -inf;

accList = zeros(size(dimList));
errList = zeros(size(dimList));
scList = zeros(size(dimList));

for d = 1:length(dimList)

    dim = dimList(d);

    features_pca = score(:, 1:dim);


    % L2归一化，适合cosine距离
    features_pca = normalize(features_pca, 2, 'norm');

    [idx, C, sumd] = kmeans(features_pca, k_cluster, ...
        'Start', 'plus', ...
        'Distance', 'cosine', ...
        'Replicates', 100, ...
        'MaxIter', 2000, ...
        'Display', 'off');

    % 计算轮廓系数
    sc = mean(silhouette(features_pca, idx));

    % 计算聚类正确率
    [acc, errRate] = calcClusterAccuracy(idx, trueLabels, classNames);

    accList(d) = acc;
    errList(d) = errRate;
    scList(d) = sc;

    fprintf('PCA维度 = %3d | 正确率 = %.2f%% | 错误率 = %.2f%% | SC = %.4f\n', ...
        dim, acc * 100, errRate * 100, sc);

    if acc > bestAcc
        bestAcc = acc;
        bestErrRate = errRate;
        bestDim = dim;
        bestIdx = idx;
        bestFeatures = features_pca;
        bestSC = sc;
    end
end

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

%% 9. 绘制 PCA维度-正确率/错误率/SC曲线
figure;

yyaxis left
plot(dimList, accList * 100, 'o-', 'LineWidth', 2);
ylabel('正确率 / %');

yyaxis right
plot(dimList, scList, 's-', 'LineWidth', 2);
ylabel('SC轮廓系数');

xlabel('PCA维度');
title('不同PCA维度下的聚类性能');
grid on;
legend('正确率', 'SC轮廓系数');

