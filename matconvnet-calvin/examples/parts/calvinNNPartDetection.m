%
% Copyright by Holger Caesar, 2016
%
% Modified by Abel Gonzalez-Garcia, 2017

% Global variables
global glDatasetFolder glFeaturesFolder;
assert(~isempty(glDatasetFolder) && ~isempty(glFeaturesFolder));

%%% Settings
% Dataset
vocYear = 2010;
trainName = 'train';
testName  = 'val';
vocName = sprintf('VOC%d', vocYear);
datasetDir = [fullfile(glDatasetFolder, vocName), '/'];


% Specify paths
outputFolder = fullfile(glFeaturesFolder, 'CNN-Models', 'Parts', vocName, sprintf('%s-baseline', vocName));
netPath = fullfile(glFeaturesFolder, 'CNN-Models', 'matconvnet', 'imagenet-caffe-alex.mat');
logFilePath = fullfile(outputFolder, 'log.txt');

% Fix randomness
randSeed = 42;
rng(randSeed);

% Setup dataset specific options and check validity
setupDataOptsPrts(vocYear, testName, datasetDir);
global DATAopts; % Database specific paths
assert(~isempty(DATAopts), 'Error: Dataset not initialized properly!');


% Task-specific
nnOpts.testFn = @testPartDetection;
nnOpts.misc.overlapNms = 0.3;
% Objectives for both parts and objects
nnOpts.derOutputs = {'objectivePrt', 1, 'objectiveObj', 1, 'regressObjectivePrt', 1, 'regressObjectiveObj', 1};

% General
nnOpts.batchSize = 2;
nnOpts.numSubBatches = nnOpts.batchSize; % 1 image per sub-batch
nnOpts.weightDecay = 5e-4;
nnOpts.momentum = 0.9;
nnOpts.numEpochs = 16;
nnOpts.learningRate = [repmat(1e-3, 12, 1); repmat(1e-4, 4, 1)];
nnOpts.misc.netPath = netPath;
nnOpts.expDir = outputFolder;
nnOpts.convertToTrain = 0; % perform explicit conversion to our architecure
nnOpts.fastRcnn = 0;
nnOpts.bboxRegress = 1;
nnOpts.gpus = 1; % for automatic selection use: SelectIdleGpu();

% Create outputFolder
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Start logging
diary(logFilePath);

%%% Setup
% Start from pretrained network
net = load(nnOpts.misc.netPath);

% Setup imdb
imdb = setupImdbPartDetection(@ImdbPartDetectionJointObjPrt,trainName, testName, net);

% Create calvinNN CNN class
% Do not transform into fast-rcnn with bbox regression
calvinn = CalvinNN(net, imdb, nnOpts);

% Perform here the conversion to part/obj architecture
calvinn.convertNetworkToPrtObjFastRcnn;

%%% Train
calvinn.train();

%%% Test
stats = calvinn.testPrtObj();

%%% Eval
evalPartAndObjectDetection(testName, stats, nnOpts);
