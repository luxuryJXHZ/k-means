
function [acc, errRate] = calcClusterAccuracy(idx, trueLabels, classNames)

    totalCorrect = 0;
    totalNum = length(idx);
    clusterNum = max(idx);

    for c = 1:clusterNum

        clusterLabels = trueLabels(idx == c);

        if isempty(clusterLabels)
            continue;
        end

        counts = zeros(length(classNames), 1);

        for j = 1:length(classNames)
            counts(j) = sum(clusterLabels == classNames(j));
        end

        totalCorrect = totalCorrect + max(counts);
    end

    acc = totalCorrect / totalNum;
    errRate = 1 - acc;
end