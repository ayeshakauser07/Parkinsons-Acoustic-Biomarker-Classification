%% CREATE SHARED 5-FOLD SUBJECT-LEVEL CV

clear;
clc;

%% Load data
data = readtable('Dataset.csv');

Y = data.Status;
Subject_ID = data.ID;

%% Get unique subjects
[~,~,g] = unique(Subject_ID);

subjects = unique(Subject_ID);

% One label per subject
subjectStatus = accumarray(g,Y,[],@mode);

%% Create 5 identical folds
rng(42);
cv5 = cvpartition(subjectStatus,'KFold',5);

%% Save the folds
save('shared_subject_cv.mat','cv5','subjects');

disp('Shared 5-fold subject-level CV created and saved.');
%% load('shared_subject_cv.mat','cv5','subjects');

for i = 1:5
    fprintf('Fold %d: %d test subjects\n',i,sum(test(cv5,i)));
end
