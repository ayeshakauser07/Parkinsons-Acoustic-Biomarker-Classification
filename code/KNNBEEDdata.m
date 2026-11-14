%% ===========================================================
% KNN Classification with ROC Curve
% Subject-Level Evaluation with Shared 5-Fold CV
% ===========================================================

clear;
clc;
close all;

%% Load Dataset
data = readtable('Dataset.csv');

disp('First 5 Rows:')
disp(head(data))

disp('Dataset Summary:')
summary(data)

%% Extract Subject ID, Features and Labels

Subject_ID = data.ID;
Y = data.Status;

X = table2array(data(:,5:end));
featureNames = data.Properties.VariableNames(5:end);

%% Load SHARED Subject-Level Cross-Validation Folds
% These folds are generated ONCE and shared by KNN, SVM and RF.

load('shared_subject_cv.mat','cv5','subjects');

%% ===========================================================
% Train-Test Split by Subject (80-20)
% ===========================================================

[~,~,g] = unique(Subject_ID);

subjectStatus = accumarray(g,Y,[],@mode);

rng(42);
cv = cvpartition(subjectStatus,'HoldOut',0.20);

trainSubjects = subjects(training(cv));
testSubjects = subjects(test(cv));

trainIdx = ismember(Subject_ID,trainSubjects);
testIdx = ismember(Subject_ID,testSubjects);

XTrain = X(trainIdx,:);
YTrain = Y(trainIdx);

XTest = X(testIdx,:);
YTest = Y(testIdx);

%% ===========================================================
% Normalize Using Training Data Only
% ===========================================================

mu = mean(XTrain);
sigma = std(XTrain);

sigma(sigma == 0) = 1;

XTrain = (XTrain - mu) ./ sigma;
XTest = (XTest - mu) ./ sigma;

%% ===========================================================
% Train KNN Model
% ===========================================================

k = 5;

knnModel = fitcknn(...
    XTrain,...
    YTrain,...
    'NumNeighbors',k,...
    'Distance','euclidean',...
    'Standardize',false);

%% ===========================================================
% Predict Labels and Scores
% ===========================================================

[YPred,scores] = predict(knnModel,XTest);

%% ===========================================================
% Accuracy
% ===========================================================

accuracy = mean(YPred == YTest)*100;

fprintf('\n====================================\n');
fprintf('KNN Accuracy = %.2f %%\n',accuracy);
fprintf('====================================\n');

%% ===========================================================
% Confusion Matrix
% ===========================================================

figure;
confusionchart(YTest,YPred);
title('KNN Confusion Matrix');

confMat = confusionmat(YTest,YPred);

disp('Confusion Matrix:')
disp(confMat)

%% ===========================================================
% Binary Metrics
% ===========================================================

if size(confMat,1) == 2

    TN = confMat(1,1);
    FP = confMat(1,2);
    FN = confMat(2,1);
    TP = confMat(2,2);

    if TP + FP > 0
        Precision = TP/(TP+FP);
    else
        Precision = 0;
    end

    if TP + FN > 0
        Recall = TP/(TP+FN);
    else
        Recall = 0;
    end

    if TN + FP > 0
        Specificity = TN/(TN+FP);
    else
        Specificity = 0;
    end

    if Precision + Recall > 0
        F1 = 2*(Precision*Recall)/(Precision+Recall);
    else
        F1 = 0;
    end

    fprintf('\nPrecision   : %.4f\n',Precision);
    fprintf('Recall      : %.4f\n',Recall);
    fprintf('Specificity : %.4f\n',Specificity);
    fprintf('F1 Score    : %.4f\n',F1);

end

%% ===========================================================
% ROC Curve
% ===========================================================

classNames = knnModel.ClassNames;
positiveClass = find(classNames == 1);

[Xroc,Yroc,~,AUC] = perfcurve(...
    YTest,...
    scores(:,positiveClass),...
    1);

figure;
plot(Xroc,Yroc,'LineWidth',2);
hold on;

plot([0 1],[0 1],'k--');

xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('KNN ROC Curve (AUC = %.3f)',AUC));

grid on;
hold off;

fprintf('\nAUC = %.4f\n',AUC);

%% ===========================================================
% 5-Fold SUBJECT-LEVEL Cross-Validation
% USING THE SHARED FOLDS
% ===========================================================

cvAccuracy = zeros(cv5.NumTestSets,1);
cvPrecision = zeros(cv5.NumTestSets,1);
cvRecall = zeros(cv5.NumTestSets,1);
cvSpecificity = zeros(cv5.NumTestSets,1);
cvF1 = zeros(cv5.NumTestSets,1);

for i = 1:cv5.NumTestSets

    fprintf('\nProcessing Fold %d...\n',i);

    %% Subject-level train/test split

    trainSubjects = subjects(training(cv5,i));
    testSubjects = subjects(test(cv5,i));

    trainIdx = ismember(Subject_ID,trainSubjects);
    testIdx = ismember(Subject_ID,testSubjects);

    XTr = X(trainIdx,:);
    XTe = X(testIdx,:);

    YTr = Y(trainIdx);
    YTe = Y(testIdx);

    %% =======================================================
    % Training-only normalization
    % =======================================================

    mu = mean(XTr);
    sigma = std(XTr);

    sigma(sigma == 0) = 1;

    XTr = (XTr - mu) ./ sigma;
    XTe = (XTe - mu) ./ sigma;

    %% =======================================================
    % KNN
    % =======================================================

    model = fitcknn(...
        XTr,...
        YTr,...
        'NumNeighbors',k,...
        'Distance','euclidean',...
        'Standardize',false);

    pred = predict(model,XTe);

    %% =======================================================
    % Metrics
    % =======================================================

    cm = confusionmat(YTe,pred);

    TN = cm(1,1);
    FP = cm(1,2);
    FN = cm(2,1);
    TP = cm(2,2);

    cvAccuracy(i) = mean(pred == YTe)*100;

    if TP + FP > 0
        cvPrecision(i) = TP/(TP+FP);
    else
        cvPrecision(i) = 0;
    end

    if TP + FN > 0
        cvRecall(i) = TP/(TP+FN);
    else
        cvRecall(i) = 0;
    end

    if TN + FP > 0
        cvSpecificity(i) = TN/(TN+FP);
    else
        cvSpecificity(i) = 0;
    end

    if cvPrecision(i) + cvRecall(i) > 0
        cvF1(i) = ...
            2*(cvPrecision(i)*cvRecall(i)) / ...
            (cvPrecision(i)+cvRecall(i));
    else
        cvF1(i) = 0;
    end

end

%% ===========================================================
% Cross-Validation Results
% ===========================================================

fprintf('\n====================================\n');
fprintf('KNN 5-Fold Subject-Level CV\n');
fprintf('====================================\n');

fprintf('Accuracy    = %.2f ± %.2f %%\n',...
    mean(cvAccuracy),std(cvAccuracy));

fprintf('Precision   = %.4f ± %.4f\n',...
    mean(cvPrecision),std(cvPrecision));

fprintf('Recall      = %.4f ± %.4f\n',...
    mean(cvRecall),std(cvRecall));

fprintf('Specificity = %.4f ± %.4f\n',...
    mean(cvSpecificity),std(cvSpecificity));

fprintf('F1 Score    = %.4f ± %.4f\n',...
    mean(cvF1),std(cvF1));

%% ===========================================================
% Feature Importance
% ===========================================================

baseline = mean(YPred == YTest);

importance = zeros(size(XTest,2),1);

rng(42);

for i = 1:size(XTest,2)

    Xp = XTest;

    Xp(:,i) = Xp(randperm(size(XTest,1)),i);

    permPred = predict(knnModel,Xp);

    importance(i) = ...
        baseline - mean(permPred == YTest);

end

%% Sort Feature Importance

[imp,idx] = sort(importance,'descend');

topN = min(15,length(imp));

disp('Top 15 Features:')

disp(table(...
    featureNames(idx(1:topN))',...
    imp(1:topN),...
    'VariableNames',{'Feature','Importance'}));

%% ===========================================================
% Feature Importance Plot
% ===========================================================

figure;

barh(imp(1:topN));

set(gca,...
    'YTick',1:topN,...
    'YTickLabel',featureNames(idx(1:topN)));

xlabel('Accuracy Decrease');
title('KNN Top 15 Feature Importance');

grid on;

%% ===========================================================
% End
% ===========================================================