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
nnOpts.testFn = @testPartDetectionwOffsetNet;
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
nnOpts.bboxRegress = 1; % no bbox-regress for Offset Net
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
ONparams.test = 0;
ONparams.trainCoeffs = 0;

imdb = setupImdbPartDetection(@ImdbOffsetNet,trainName, testName, net, ONparams);

% Create calvinNN CNN class
% Do not transform into fast-rcnn with bbox regression
calvinn = CalvinNN(net, imdb, nnOpts);
% Perform here the conversion to part/obj architecture
calvinn.convertNetworkToOffsetNet('numObjClasses',imdb.numClassesObj,...
    'numPrtClasses',imdb.numClassesPrt,'numOuts',numOuts);


t1 = tic();
%%% Train
calvinn.train();
times(3,1) = toc(t1);



%%% Test
netObjAppClsPath = fullfile(glFeaturesFolder, 'CNN-Models', 'Parts', vocName, sprintf('%s-ObjAppCls', vocName), 'net-epoch-16.mat');

% Load ObjAppCls network so we can add OffsetNet to it
netObjAppCls = load(netObjAppClsPath);
netObjAppCls = netObjAppCls.net;

% Setup again imdb --> for testing we need all images (without parts too)
ONparams.test = 1;
[imdb, idxPartGlobal2idxClass] = setupImdbPartDetection(@ImdbOffsetNet, trainName, testName, net, ONparams);


nnOpts.misc.numOuts = numOuts;
nnOpts.misc.idxPartGlobal2idxClass = idxPartGlobal2idxClass(2:end);

calvinnObjAppCls = CalvinNN(netObjAppCls, imdb, nnOpts);

calvinnObjAppCls.mergeObjAppClswOffsetNet('offsetNet', calvinn.net, 'numOuts', max(numOuts));
t2 = tic;
stats = calvinnObjAppCls.test;
times(3,2) = toc(t2);

save([nnOpts.expDir 'stats.mat'], 'stats','-v7.3');


%%% Train coefficients if necessary
if trainCoeffs
    ONparams.trainCoeffs = 1;
    imdb = setupImdbPartDetection(@ImdbOffsetNet, trainName, testName, net, ONparams);
   
    calvinnObjAppCls = CalvinNN(netObjAppCls, imdb, nnOpts);
    calvinnObjAppCls.mergeObjAppClswOffsetNet('offsetNet', calvinn.net, 'numOuts', max(numOuts));
    
    statsTr = calvinnObjAppCls.test;
    
    CM = trainCoefficients('train',statsTr);
    
else
    % Load coefficients
    trash = load([nnOpts.expDir 'coeffs.mat']);
    CM = trash.CM;
    clear trash;
end

nnOpts.CM = CM;
evalPartAndObjectDetection(testName, stats, nnOpts);

