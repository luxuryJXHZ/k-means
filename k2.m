clc; clear; close all;

%% 1. 图像路径
imgDir = fullfile(pwd, 'flowers');
outDir = fullfile(pwd, 'kmeans_result1');

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

class0Dir = fullfile(outDir, 'cluster_0');
class1Dir = fullfile(outDir, 'cluster_1');

if ~exist(class0Dir, 'dir'), mkdir(class0Dir); end
if ~exist(class1Dir, 'dir'), mkdir(class1Dir); end

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

featureLayer = 'global_average_pooling2d_1';

features = [];

%% 3. 提取特征
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

    fprintf('处理第 %d / %d 张\n', i, numImages);
end




%% 6. K-means 聚类
cluster_num = 5;

[idx, center] = kmeans(features, cluster_num, ...
    'Replicates', 5, ...
    'MaxIter', 300);


%% 7. 保存分类结果
for i = 1:numImages
    srcPath = fullfile(files(i).folder, files(i).name);

    if idx(i) == 1
        dstPath = fullfile(class0Dir, files(i).name);
    else
        dstPath = fullfile(class1Dir, files(i).name);
    end

    copyfile(srcPath, dstPath);
end

fprintf('\n聚类完成！结果已保存到：%s\n', outDir);

%% 8. 计算并打印 SC 轮廓系数
sc = mean(silhouette(features, idx));

fprintf('\nK-means 聚类完成！\n');
fprintf('SC 轮廓系数：%.4f\n', sc);
fprintf('cluster_0 图像数量：%d\n', sum(idx == 1));
fprintf('cluster_1 图像数量：%d\n', sum(idx == 2));