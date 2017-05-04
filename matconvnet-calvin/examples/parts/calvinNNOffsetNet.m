%
% Copyright by Holger Caesar, 2016
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
outputFolder = fullfile(glFeaturesFolder, 'CNN-Models', 'Parts', vocName, sprintf('%s-OffsetNet', vocName));
% Initialize network with baseline
netPath = fullfile(glFeaturesFolder, 'CNN-Models', 'Parts', vocName, sprintf('%s-baseline', vocName), 'net-epoch-16.mat');
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
nnOpts.derOutputs = {'dispWindows1Objective', 1};

% Get number of outputs per part class
numOuts = getNumOuts();

for idxNumOuts = 2:max(numOuts)
    nnOpts.derOutputs{end+1} = ['dispWindows' num2str(idxNumOuts) 'Objective'];
    nnOpts.derOutputs{end+1} = 1;
end
nnOpts.derOutputs{end+1} =  'presenceObjective';
nnOpts.derOutputs{end+1} =  1;

% General
nnOpts.batchSize = 2;
nnOpts.numSubBatches = nnOpts.batchSize; % 1 image per sub-batch
nnOpts.weightDecay = 5e-4;
nnOpts.momentum = 0.9;
nnOpts.numEpochs = 16;
% Smaller LR for offset net
nnOpts.learningRate = [repmat(1e-4, 12, 1); repmat(1e-5, 4, 1)];
nnOpts.misc.netPath = netPath;
nnOpts.expDir = outputFolder;
nnOpts.convertToTrain = 0; % perform explicit conversion to our architecure
nnOpts.fastRcnn = 0;
nnOpts.bboxRegress = 0; % no bbox-regress for Offset Net
nnOpts.gpus = 1; % for automatic selection use: SelectIdleGpu();

% Create outputFolder
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Start logging
diary(logFilePath);

%%% Setup
% Start from pretrained network
netBaseline = load(nnOpts.misc.netPath);
net = netBaseline.net;

ONparams.numOuts =  numOuts;  
% Object classes without parts (here for PASCAL-Part, change if other data)
ONparams.idxObjClassRem = [4 9 11 18];

imdb = setupImdbPartDetection(@ImdbOffsetNet,trainName, testName, net, ONparams);

% Create calvinNN CNN class
% Do not transform into fast-rcnn with bbox regression
calvinn = CalvinNN(net, imdb, nnOpts);

% Perform here the conversion to part/obj architecture
calvinn.convertNetworkToOffsetNet('numObjClasses',imdb.numClassesObj,...
    'numPrtClasses',imdb.numClassesPrt,'numOuts',numOuts);

%%% Train
calvinn.train();

%%% Test
netObjAppClsPath = fullfile(glFeaturesFolder, 'CNN-Models', 'Parts', vocName, sprintf('%s-ObjAppCls', vocName), 'net-epoch-16.mat');
netObjAppCls = load(netObjAppClsPath);
netObjAppCls = netObjAppCls.net;


