clc; clear; close all;

%% 1. 图像路径
imgDir = fullfile(pwd, 'fl');
outDir = fullfile(pwd, 'kmeans_result1');

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
featureLayer = 'Logits';

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

%% 4. 特征归一化
% 先做一次 PCA（只做一次！）
features_norm = normalize(features, 2);
[~, score] = pca(features_norm);

K= 5;

sc1 = zeros(1,100);
errorSum = zeros(1,100);

for dim = 1:100
    
    features_pca = score(:, 1:dim);
    
    [idx, C] = kmeans(features_pca, K, ...
        'Start', 'plus', ...
        'Distance', 'cosine', ...
        'Replicates', 30, ...
        'MaxIter', 1000);
    
    sc = mean(silhouette(features_pca, idx));
    sc1(dim) = sc;

    outDir = fullfile(pwd, 'kmeans_result1');

    if exist(outDir, 'dir')
        rmdir(outDir, 's');
    end

    mkdir(outDir);

    cluster_num = max(idx);
    clusterDirs = cell(cluster_num, 1);

    for c = 1:cluster_num
        clusterDirs{c} = fullfile(outDir, ['cluster_', num2str(c)]);
        mkdir(clusterDirs{c});
    end

    for n = 1:numImages
        srcPath = fullfile(files(n).folder, files(n).name);
        cluster_id = idx(n);
        dstPath = fullfile(clusterDirs{cluster_id}, files(n).name);
        copyfile(srcPath, dstPath);
    end

    for c = 1:cluster_num
        result = count_cluster_general(clusterDirs{c});
        errorSum(dim) = errorSum(dim) + result.errorNum;
    end

    fprintf('维度 = %d, SC = %.4f\n', dim, sc);
    fprintf('错误率 = %.4f\n', errorSum(dim) / numImages);

    rmdir(outDir, 's');
end

% 循环结束后画图
dims = 1:100;
errorRate = errorSum / numImages * 100;

figure;

yyaxis left
plot(dims, sc1, 'o-', 'LineWidth', 1.5);
ylabel('轮廓系数 SC');

yyaxis right
plot(dims, errorRate, 's-', 'LineWidth', 1.5);
ylabel('错误率 / %');

xlabel('PCA维度');
title('不同PCA维度下的SC与错误率变化');
legend('SC', '错误率', 'Location', 'best');
grid on;

