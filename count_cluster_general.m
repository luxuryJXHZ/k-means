                function result = count_cluster_general(folderPath)

files = [ ...
    dir(fullfile(folderPath, '*.jpg')); ...
    dir(fullfile(folderPath, '*.png')); ...
    dir(fullfile(folderPath, '*.jpeg')) ...
];

numFiles = length(files);

if numFiles == 0
    fprintf('文件夹为空：%s\n', folderPath);
    result = [];
    return;
end

labels = strings(numFiles,1);

%% 1. 提取前缀作为类别
for i = 1:numFiles
    name = lower(files(i).name);

    token = regexp(name, '^[a-zA-Z]+', 'match');

    if ~isempty(token)
        labels(i) = string(token{1});
    else
        labels(i) = "unknown";
    end
end

%% 2. 统计每类数量
[unique_labels, ~, idx] = unique(labels);
counts = accumarray(idx, 1);

%% 3. 找出该文件夹的主类别
[maxCount, maxIdx] = max(counts);
mainClass = unique_labels(maxIdx);

errorNum = numFiles - maxCount;
errorRate = errorNum / numFiles;

%% 4. 输出
fprintf('\n===== 文件夹统计：%s =====\n', folderPath);

for i = 1:length(unique_labels)
    fprintf('%s: %d\n', unique_labels(i), counts(i));
end

fprintf('总数量: %d\n', numFiles);
fprintf('该cluster判定类别: %s\n', mainClass);
fprintf('错分数量: %d\n', errorNum);
fprintf('错误率: %.2f%%\n', errorRate * 100);

%% 5. 返回结果
result.folderPath = folderPath;
result.mainClass = mainClass;
result.totalNum = numFiles;
result.errorNum = errorNum;
result.errorRate = errorRate;
result.labels = unique_labels;
result.counts = counts;

end