function convertNetworkToOffsetNet(obj, varargin)
%
% Modify network for Offset Net
%
% Copyright by Holger Caesar, 2015
% Updated by Jasper Uijlings:
%  - Extra flexibility and possible bounding box regression
%  - Added instanceWeights to loss layer
% 
% Modified by Abel Gonzalez-Garcia, 2016

% Initial settings
p = inputParser;
addParameter(p, 'lastConvPoolName', 'pool5');
addParameter(p, 'firstFCName', 'fc6');
addParameter(p, 'secondFCName', 'fc7');
addParameter(p, 'finalFCName', 'fc8');
addParameter(p, 'numObjClasses', '21');
addParameter(p, 'numPrtClasses', '106');
addParameter(p, 'numOuts','0');
parse(p, varargin{:});

lastConvPoolName = p.Results.lastConvPoolName;
firstFCName = p.Results.firstFCName;
secondFCName =  p.Results.secondFCName;
finalFCName = p.Results.finalFCName;
numObjClasses = p.Results.numObjClasses;
numPrtClasses = p.Results.numPrtClasses;
numOuts = p.Results.numOuts;

maxNewLayers = max(numOuts);


%% Prepare inputs of regression layer for displaced windows

finalFCLayerIdx = obj.net.getLayerIndex([finalFCName 'Obj']);
inputVars = obj.net.layers(finalFCLayerIdx).inputs;

finalFCLayerSize = size(obj.net.params(obj.net.layers(finalFCLayerIdx).paramIndexes(1)).value);

regressLayerSize = [1 1 finalFCLayerSize(3) 4*(numPrtClasses-1)];

%% Remove all unnecessary layers

% Re-use the object class branch for Offset Net
% Part layers
for idxLayer = size(obj.net.layers,2):-1:1
   if strcmp(obj.net.layers(idxLayer).name(end-2:end),'Prt')
       removeLayer(obj.net, obj.net.layers(idxLayer).name);
   end
end

% Also remove unnecessary object layers
removeLayer(obj.net,[finalFCName 'Obj']);
removeLayer(obj.net,'softmaxlossObj');
removeLayer(obj.net,[finalFCName 'regressObj']);
removeLayer(obj.net,'regressLossObj');

% Rename input of RoiPooling
renameVar(obj.net,'boxesObj','objDets');

%% Add new regression layers for displaced windows (maxNewLayers, parallel)

for idxNewLayer = 1:maxNewLayers
    regressName = [finalFCName 'DispWindows' num2str(idxNewLayer)];
    obj.net.addLayer(regressName, dagnn.Conv('size', regressLayerSize) , inputVars,...
        {['dispWindows' num2str(idxNewLayer) 'Scores']},  {['dispWindows' num2str(idxNewLayer) 'f'], ['dispWindows' num2str(idxNewLayer) 'b']});
    regressIdx = obj.net.getLayerIndex(regressName);
    newParams = obj.net.layers(regressIdx).block.initParams();
    obj.net.params(obj.net.layers(regressIdx).paramIndexes(1)).value = newParams{1} / std(newParams{1}(:)) * 0.001; % Girshick initialization with std of 0.001
    obj.net.params(obj.net.layers(regressIdx).paramIndexes(2)).value = newParams{2};
    
    % Add loss
    obj.net.addLayer(['dispWindows'  num2str(idxNewLayer) 'Loss'], dagnn.LossRegress('loss', 'Smooth', 'smoothMaxDiff', 1), ...
        {['dispWindows'  num2str(idxNewLayer) 'Scores'], ['dispWindows'  num2str(idxNewLayer) 'Targets'] , 'instanceWeights'},...
        ['dispWindows'  num2str(idxNewLayer) 'Objective']);
end

%% Add presence layer

presenceLayerSize = [1 1  finalFCLayerSize(3) sum(numOuts)];

addLayer(obj.net, [finalFCName 'presence'], dagnn.Conv('size', presenceLayerSize), inputVars,...
    'presenceScores',{[finalFCName 'presencef'],[finalFCName 'presenceb']});

presenceIdx = obj.net.getLayerIndex([finalFCName 'presence']);
newParams = obj.net.layers(presenceIdx).block.initParams();

% Initialize parameters
obj.net.params(obj.net.layers(presenceIdx).paramIndexes(1)).value = newParams{1} / std(newParams{1}(:)) * 0.001; % Girshick initialization with std of 0.001
obj.net.params(obj.net.layers(presenceIdx).paramIndexes(2)).value = newParams{2};

obj.net.addLayer('presenceLogLoss', dagnn.LossWeighted('loss', 'logistic'), ...
    { obj.net.layers(presenceIdx).outputs{1}, 'presenceTargets', 'instanceWeights'}, 'presenceObjective');


sortLayers(obj.net);

%%% Set correct learning rates and biases (Girshick style)
if obj.nnOpts.fastRcnnParams
    % Biases have learning rate of 2 and no weight decay
    for lI = 1 : length(obj.net.layers)
        if isa(obj.net.layers(lI).block, 'dagnn.Conv')
            biasI = obj.net.layers(lI).paramIndexes(2);
            obj.net.params(biasI).learningRate = 2;
            obj.net.params(biasI).weightDecay = 0;
        end
    end
    
    conv1I = obj.net.getLayerIndex('conv1'); % AlexNet-style networks
    if isnan(conv1I)
        conv1I = obj.net.getLayerIndex('conv1_1'); % VGG-16 style networks
    end
    obj.net.params(obj.net.layers(conv1I).paramIndexes(1)).learningRate = 0;
    obj.net.params(obj.net.layers(conv1I).paramIndexes(2)).learningRate = 0;
end
