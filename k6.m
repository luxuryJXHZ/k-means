clc; clear; close all;

%% 1. 图像路径
imgDir = fullfile(pwd, 'dogandcat');
outDir = fullfile(pwd, 'kmeans_result');
    if exist(outDir, 'dir')
        rmdir(outDir, 's');
    end

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

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

% 推荐使用全局平均池化层特征
featureLayer = 'global_average_pooling2d_1';

features = [];

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
    feat = single(feat);
    features(i, :) = feat;

    fprintf('正在处理第 %d / %d 张：%s\n', ...
        i, numImages, files(i).name);
end

%% 4. K-means 聚类
k = 2;

[idx, C, sumd] = kmeans(features, k, ...
    'Replicates', 5, ...
    'MaxIter', 300, ...
    'Display', 'final');
%% 5. 使用训练中心 C 验证测试集
testDir = fullfile(pwd, 'catdogtest');

testFiles = [ ...
    dir(fullfile(testDir, '*.jpg')); ...
    dir(fullfile(testDir, '*.png')); ...
    dir(fullfile(testDir, '*.jpeg')) ...
];

numTest = numel(testFiles);
fprintf('\n共读取到 %d 张测试图像。\n', numTest);
cluster_num = max(idx);   % 自动获取聚类类别数
%% 先确定 cluster_1 / cluster_2 分别对应 cat 还是 dog
% 根据训练集文件名判断每张训练图的真实标签
trainLabels = strings(numImages, 1);

for i = 1:numImages
    name = lower(files(i).name);

    if startsWith(name, 'cat')
        trainLabels(i) = "cat";
    elseif startsWith(name, 'dog')
        trainLabels(i) = "dog";
    else
        trainLabels(i) = "unknown";
    end
end

% 统计每个聚类中 cat / dog 哪个更多
clusterName = strings(cluster_num, 1);

for c = 1:cluster_num
    labelsInCluster = trainLabels(idx == c);

    catNum = sum(labelsInCluster == "cat");
    dogNum = sum(labelsInCluster == "dog");

    if catNum >= dogNum
        clusterName(c) = "cat";
    else
        clusterName(c) = "dog";
    end
end

fprintf('\n聚类中心类别映射：\n');
for c = 1:cluster_num
    fprintf('cluster_%d -> %s\n', c, clusterName(c));
end

%% 测试集分类
correct = 0;

for i = 1:numTest
    imgPath = fullfile(testFiles(i).folder, testFiles(i).name);
    img = imread(imgPath);

    if size(img, 3) == 1
        img = cat(3, img, img, img);
    end

    img = imresize(img, inputSize(1:2));

    feat = activations(net, img, featureLayer, ...
        'OutputAs', 'rows');
    feat = single(feat);
    % 计算测试特征到两个训练中心的距离
    dist = pdist2(feat, C);

    % 找最近的聚类中心
    [~, predCluster] = min(dist);

    % 得到预测类别
    predLabel = clusterName(predCluster);

    % 根据文件名得到真实类别
    fileName = lower(testFiles(i).name);

    if startsWith(fileName, 'cat')
        trueLabel = "cat";
    elseif startsWith(fileName, 'dog')
        trueLabel = "dog";
    else
        trueLabel = "unknown";
    end

    % 判断是否正确
    if predLabel == trueLabel
        correct = correct + 1;
    end
end

accuracy = correct / numTest * 100;

fprintf('\n测试集验证完成！\n');
fprintf('测试集总数：%d\n', numTest);
fprintf('分类正确数：%d\n', correct);
fprintf('分类准确率：%.2f%%\n', accuracy);