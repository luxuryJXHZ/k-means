function C = farthestInit(X, k)

    n = size(X, 1);

    % 先选离整体均值最远的样本作为第一个中心
    meanX = mean(X, 1);
    d0 = pdist2(X, meanX, 'cosine');
    [~, firstIdx] = max(d0);

    centerIdx = zeros(k, 1);
    centerIdx(1) = firstIdx;

    % 逐个选取距离已有中心最远的样本
    for c = 2:k
        D = pdist2(X, X(centerIdx(1:c-1), :), 'cosine');

        % 每个样本到最近已有中心的距离
        minD = min(D, [], 2);

        % 不重复选中心
        minD(centerIdx(1:c-1)) = -inf;

        [~, nextIdx] = max(minD);
        centerIdx(c) = nextIdx;
    end

    C = X(centerIdx, :);
end