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

%% 6. 计算并打印 SC 轮廓系数
sc = mean(silhouette(features, idx));

fprintf('\nK-means 聚类完成！\n');
fprintf('SC 轮廓系数：%.4f\n', sc);
fprintf('cluster_0 图像数量：%d\n', sum(idx == 1));
fprintf('cluster_1 图像数量：%d\n', sum(idx == 2));