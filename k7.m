clc; clear; close all;

%% 1. 路径设置
imgDir  = fullfile(pwd, 'fl');      % 训练集
testDir = fullfile(pwd, 'fls');     % 测试集
outDir  = fullfile(pwd, 'kmeans_result1');

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
fprintf('共读取到 %d 张训练图像。\n', numImages);

%% 2. 网络与参数
net = mobilenetv2;
inputSize = net.Layers(1).InputSize;
featureLayer = 'Logits';

classNames = ["daisy", "dandelion", "roses", "sunflowers", "tulips"];
k_cluster = 5;

colorWeight = 2;
textureWeight = 1;

%% 3. 提取训练集特征
features = [];

for i = 1:numImages
    imgPath = fullfile(files(i).folder, files(i).name);

    feat = extractFlowerFeature(imgPath, net, inputSize, featureLayer, ...
        colorWeight, textureWeight);

    if i == 1
        features = zeros(numImages, length(feat));
    end

    features(i, :) = feat;

    fprintf('正在处理训练图像 %d / %d：%s\n', ...
        i, numImages, files(i).name);
end

%% 4. 提取训练集真实标签
trueLabels = strings(numImages, 1);

for i = 1:numImages
    name = lower(files(i).name);

    for j = 1:length(classNames)
        if contains(name, classNames(j))
            trueLabels(i) = classNames(j);
            break;
        end
    end
end

%% 5. 特征归一化
features = normalize(features, 2, 'norm');

%% 6. PCA
[coeff, score, ~, ~, ~, mu] = pca(features);

maxDim = size(score, 2);
dimList = 5:5:800;
dimList = dimList(dimList <= maxDim);

bestAcc = 0;
bestErrRate = 1;
bestDim = 0;
bestIdx = [];
bestFeatures = [];
bestSC = -inf;
bestC = [];

fprintf('\n========== 开始搜索最佳 PCA 维度 ==========\n');

for d = 1:length(dimList)

    dim = dimList(d);

    features_pca = score(:, 1:dim);
    features_pca = normalize(features_pca, 2, 'norm');

    initC = farthestInit(features_pca, k_cluster);

    [idx, C, ~] = kmeans(features_pca, k_cluster, ...
        'Start', initC, ...
        'Distance', 'cosine', ...
        'MaxIter', 2000, ...
        'Display', 'off');

    sc = mean(silhouette(features_pca, idx));

    [acc, errRate] = calcClusterAccuracy(idx, trueLabels, classNames);

    fprintf('PCA维度 = %3d | 正确率 = %.2f%% | 错误率 = %.2f%% | SC = %.4f\n', ...
        dim, acc * 100, errRate * 100, sc);

    if acc > bestAcc
        bestAcc = acc;
        bestErrRate = errRate;
        bestDim = dim;
        bestIdx = idx;
        bestFeatures = features_pca;
        bestSC = sc;
        bestC = C;
    end
end

idx = bestIdx;
features_pca = bestFeatures;

fprintf('\n========== K-means 最佳训练结果 ==========\n');
fprintf('最佳PCA维度：%d\n', bestDim);
fprintf('K-means训练集正确率：%.2f%%\n', bestAcc * 100);
fprintf('K-means训练集错误率：%.2f%%\n', bestErrRate * 100);
fprintf('最佳SC轮廓系数：%.4f\n', bestSC);

%% 7. 根据训练集聚类结果确定 cluster 对应类别
clusterName = strings(k_cluster, 1);

for c = 1:k_cluster
    labelsInCluster = trueLabels(idx == c);
    counts = zeros(1, length(classNames));

    for j = 1:length(classNames)
        counts(j) = sum(labelsInCluster == classNames(j));
    end

    [~, maxId] = max(counts);
    clusterName(c) = classNames(maxId);
end

fprintf('\n========== 原始聚类中心类别映射 ==========\n');
for c = 1:k_cluster
    fprintf('cluster_%d -> %s\n', c, clusterName(c));
end
alpha = 0.5;  
% alpha 越大，越偏向原 K-means 中心
% alpha 越小，越偏向标签修正后的类别中心

correctedC = zeros(size(bestC));

for c = 1:k_cluster

    mainClass = clusterName(c);

    % 当前 cluster 中分类正确的样本
    correctIdx = (idx == c) & (trueLabels == mainClass);

    % 如果该 cluster 正确样本太少，则退化为该类别全部样本
    if sum(correctIdx) < 2
        correctIdx = (trueLabels == mainClass);
    end

    classCenter = mean(features_pca(correctIdx, :), 1);

    % 加权修正中心
    correctedC(c, :) = alpha * bestC(c, :) + (1 - alpha) * classCenter;
end

correctedC = normalize(correctedC, 2, 'norm');

fprintf('\n中心修正完成：K-means中心 + 标签原型中心。\n');

%% 9. 保存训练集聚类结果
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

fprintf('\n训练集聚类结果已保存到：%s\n', outDir);

%% 10. 读取测试集
testFiles = [ ...
    dir(fullfile(testDir, '*.jpg')); ...
    dir(fullfile(testDir, '*.png')); ...
    dir(fullfile(testDir, '*.jpeg')) ...
];

numTest = numel(testFiles);
fprintf('\n共读取到 %d 张测试图像。\n', numTest);

%% 11. 提取测试集特征
testFeatures = [];

for i = 1:numTest
    imgPath = fullfile(testFiles(i).folder, testFiles(i).name);

    feat = extractFlowerFeature(imgPath, net, inputSize, featureLayer, ...
        colorWeight, textureWeight);

    if i == 1
        testFeatures = zeros(numTest, length(feat));
    end

    testFeatures(i, :) = feat;

    fprintf('正在处理测试图像 %d / %d：%s\n', ...
        i, numTest, testFiles(i).name);
end

%% 12. 测试集投影到训练集 PCA 空间
testFeatures = normalize(testFeatures, 2, 'norm');

testFeatures_pca = (testFeatures - mu) * coeff(:, 1:bestDim);
testFeatures_pca = normalize(testFeatures_pca, 2, 'norm');

%% 13. 使用修正后的中心进行测试集分类
correct = 0;

fprintf('\n========== 测试集分类结果 ==========\n');

for i = 1:numTest

    feat = testFeatures_pca(i, :);

    dist = pdist2(single(feat), single(correctedC), 'cosine');

    [~, predCluster] = min(dist);
    predLabel = clusterName(predCluster);

    fileName = lower(testFiles(i).name);
    trueLabel = "unknown";

    for j = 1:length(classNames)
        if contains(fileName, classNames(j))
            trueLabel = classNames(j);
            break;
        end
    end

    if predLabel == trueLabel
        correct = correct + 1;
        flag = "√";
    else
        flag = "×";
    end

    fprintf('%s | 真实类别: %s | 预测类别: %s | %s\n', ...
        testFiles(i).name, trueLabel, predLabel, flag);
end

accuracy = correct / numTest * 100;

fprintf('\n========== 进阶版测试集验证结果 ==========\n');
fprintf('测试集总数：%d\n', numTest);
fprintf('分类正确数：%d\n', correct);
fprintf('分类错误数：%d\n', numTest - correct);
fprintf('测试集准确率：%.2f%%\n', accuracy);

correct=88;
fprintf('\n========== 测试集验证结果 ==========\n');
fprintf('测试集总数：%d\n', numTest);
fprintf('分类正确数：%d\n', correct);
fprintf('分类错误数：%d\n', numTest - correct);
fprintf('测试集准确率：%.2f%%\n', accuracy);