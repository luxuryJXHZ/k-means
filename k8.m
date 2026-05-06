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
featureLayer = 'global_average_pooling2d_1';

classNames = ["daisy", "dandelion", "roses", "sunflowers", "tulips"];
k_cluster = 5;

colorWeight = 2;
textureWeight = 1;

%% 3. 提取训练集特征
features = [];

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
k = 5;

[idx, C, sumd] = kmeans(features, k, ...
    'Replicates', 5, ...
    'MaxIter', 300, ...
    'Display', 'final');
%% 5. 保存分类结果
%% 自动创建聚类结果文件夹
outDir = fullfile(pwd, 'kmeans_result');

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

cluster_num = max(idx);   % 自动获取聚类类别数

clusterDirs = cell(cluster_num, 1);

for k = 1:cluster_num
    clusterDirs{k} = fullfile(outDir, ['cluster_', num2str(k)]);
    
    if ~exist(clusterDirs{k}, 'dir')
        mkdir(clusterDirs{k});
    end
end

%% 按聚类结果复制图像
for i = 1:numImages
    srcPath = fullfile(files(i).folder, files(i).name);

    cluster_id = idx(i);
    dstPath = fullfile(clusterDirs{cluster_id}, files(i).name);

    copyfile(srcPath, dstPath);
end

fprintf('分类结果已保存到：%s\n', outDir);

fprintf('\n聚类完成！结果已保存到：%s\n', outDir);

for k = 1:cluster_num
    count_cluster_general(clusterDirs{k});
end

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

%% 11. 提取测试集特征并按最近聚类中心分类
correct = 0;

fprintf('\n========== 测试集分类结果 ==========\n');

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

    % 只计算测试样本到训练得到的聚类中心 C 的距离
    dist = pdist2(feat, single(C));

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

fprintf('\n========== 测试集验证结果 ==========\n');
fprintf('测试集总数：%d\n', numTest);
fprintf('分类正确数：%d\n', correct);
fprintf('分类错误数：%d\n', numTest - correct);
fprintf('测试集准确率：%.2f%%\n', accuracy);

