function convertNetworkToPrtObjFastRcnn(obj, varargin)
%
% Modify network for Fast R-CNN's ROI pooling.
%
% Copyright by Holger Caesar, 2015
% Updated by Jasper Uijlings:
%  - Extra flexibility and possible bounding box regression
%  - Added instanceWeights to loss layer

% Initial settings
p = inputParser;
addParameter(p, 'lastConvPoolName', 'pool5');
addParameter(p, 'firstFCName', 'fc6');
addParameter(p, 'secondFCName', 'fc7');
addParameter(p, 'finalFCName', 'fc8');
parse(p, varargin{:});

lastConvPoolName = p.Results.lastConvPoolName;
firstFCName = p.Results.firstFCName;
secondFCName =  p.Results.secondFCName;
finalFCName = p.Results.finalFCName;

% Make these parameters
numObjClasses = 21;
numPrtClasses = 106;

% Rename input
if ~isnan(obj.net.getVarIndex('x0'))
    % Input variable x0 is renamed to input
    obj.net.renameVar('x0', 'input');
else
    % Input variable already has the correct name
    assert(~isnan(obj.net.getVarIndex('input')));
end


% Remove unused layers from pre-trained network
removeLayer(obj.net,'prob');
removeLayer(obj.net,finalFCName);

% Get number of last variable
numLastVar = str2double(obj.net.vars(end).name(2:end));

%%% Replace pooling layer of last convolution layer with roiPooling for
%%% objects
lastConvPoolIdx = obj.net.getLayerIndex(lastConvPoolName);
assert(~isnan(lastConvPoolIdx));
roiPoolName = ['roi', lastConvPoolName 'Obj'];
firstFCIdx = obj.net.layers(lastConvPoolIdx).outputIndexes;
assert(length(firstFCIdx) == 1);
roiPoolSize = obj.net.layers(firstFCIdx).block.size(1:2);
roiPoolBlock = dagnn.RoiPooling('poolSize', roiPoolSize);
replaceLayer(obj.net, lastConvPoolName, roiPoolName, roiPoolBlock, {'oriImSize', 'boxesObj'}, {'roiPoolMaskObj'});

renameLayer(obj.net, firstFCName, [firstFCName 'Obj']);
% Leave original names of params, matconvnet-calvin uses matconvnet beta2,
% which does not have renameParam function
% renameParam(obj.net, [firstFCName 'f'], [firstFCName 'Objf']);
% renameParam(obj.net, [firstFCName 'b'], [firstFCName 'Objb']);

renameLayer(obj.net, ['relu' firstFCName(end)], ['relu' firstFCName(end) 'Obj']);
% renameLayer(obj.net, ['dropout' firstFCName(end)], ['dropout' firstFCName(end) 'Obj']);

insertLayer(obj.net, ['relu' firstFCName(end) 'Obj'], secondFCName, 'dropout6Obj', dagnn.DropOut());
% Increment last var counter as insertLayer increments it
numLastVar = numLastVar + 1;

renameLayer(obj.net, secondFCName, [secondFCName 'Obj']);
% renameParam(obj.net, [secondFCName 'f'], [secondFCName 'Objf']);
% renameParam(obj.net, [secondFCName 'b'], [secondFCName 'Objb']);

renameLayer(obj.net, ['relu' secondFCName(end)], ['relu' secondFCName(end) 'Obj']);
addLayer(obj.net, ['dropout' secondFCName(end) 'Obj'],dagnn.DropOut(),...
    obj.net.layers(obj.net.getLayerIndex(['relu' secondFCName(end) 'Obj'])).outputs{1},...
    ['x' num2str(numLastVar + 1)]);
numLastVar = numLastVar + 1;


addLayer(obj.net,[finalFCName 'Obj'], dagnn.Conv('size',[1 1 4096 numObjClasses]),...
     obj.net.layers(obj.net.getLayerIndex(['dropout' secondFCName(end) 'Obj'])).outputs,...
     ['x' num2str(numLastVar +1)], {[finalFCName 'Objf'],[finalFCName 'Objb']});
numLastVar = numLastVar + 1;

% Initialize parameters
finalFCIdx = obj.net.getLayerIndex([finalFCName 'Obj']);
newParams = obj.net.layers(finalFCIdx).block.initParams();
% Initialize parameters
obj.net.params(obj.net.layers(finalFCIdx).paramIndexes(1)).value = newParams{1} / std(newParams{1}(:)) * 0.001; % Girshick initialization with std of 0.001
obj.net.params(obj.net.layers(finalFCIdx).paramIndexes(2)).value = newParams{2};


%%% Now add part branch

% First input of RoiPooling is the same
lastConvPoolIdx = obj.net.getLayerIndex(['roi', lastConvPoolName 'Obj']);

roiPoolName = ['roi', lastConvPoolName 'Prt'];
roiPoolBlock = dagnn.RoiPooling('poolSize', roiPoolSize);
addLayer(obj.net, roiPoolName, roiPoolBlock,...
    {obj.net.layers(lastConvPoolIdx).inputs{1},'oriImSize', 'boxesPrt'},...
    {['x' num2str(numLastVar +1)], 'roiPoolMaskPrt'});
numLastVar = numLastVar + 1;

addLayer(obj.net, [firstFCName 'Prt'],...
    dagnn.Conv('size',obj.net.layers(obj.net.getLayerIndex([firstFCName 'Obj'])).block.size),...
    ['x' num2str(numLastVar)], ['x' num2str(numLastVar+1)], {[firstFCName 'Prtf'],[firstFCName 'Prtb']});
numLastVar = numLastVar + 1;

% Init params with the pre-trained network params (now in obj branch)   
idxParamsObj = obj.net.layers(obj.net.getLayerIndex([firstFCName 'Obj'])).paramIndexes;
idxParamsPrt = obj.net.layers(obj.net.getLayerIndex([firstFCName 'Prt'])).paramIndexes;
obj.net.params(idxParamsPrt(1)).value = obj.net.params(idxParamsObj(1)).value;
obj.net.params(idxParamsPrt(2)).value = obj.net.params(idxParamsObj(2)).value;

addLayer(obj.net, ['relu' firstFCName(end) 'Prt'], dagnn.ReLU,...
    ['x' num2str(numLastVar)], ['x' num2str(numLastVar+1)],{});
numLastVar = numLastVar + 1;

addLayer(obj.net, ['dropout' firstFCName(end) 'Prt'],dagnn.DropOut(),...
    ['x' num2str(numLastVar)],['x' num2str(numLastVar + 1)]);
numLastVar = numLastVar + 1;


addLayer(obj.net, [secondFCName 'Prt'],...
    dagnn.Conv('size',obj.net.layers(obj.net.getLayerIndex([secondFCName 'Obj'])).block.size),...
    ['x' num2str(numLastVar)], ['x' num2str(numLastVar+1)], {[secondFCName 'Prtf'],[secondFCName 'Prtb']});
numLastVar = numLastVar + 1;

% Init params with the pre-trained network params (now in obj branch)   
idxParamsObj = obj.net.layers(obj.net.getLayerIndex([secondFCName 'Obj'])).paramIndexes;
idxParamsPrt = obj.net.layers(obj.net.getLayerIndex([secondFCName 'Prt'])).paramIndexes;
obj.net.params(idxParamsPrt(1)).value = obj.net.params(idxParamsObj(1)).value;
obj.net.params(idxParamsPrt(2)).value = obj.net.params(idxParamsObj(2)).value;


addLayer(obj.net, ['relu' secondFCName(end) 'Prt'], dagnn.ReLU,...
    ['x' num2str(numLastVar)], ['x' num2str(numLastVar+1)],{});
numLastVar = numLastVar + 1;

addLayer(obj.net, ['dropout' secondFCName(end) 'Prt'],dagnn.DropOut(),...
    ['x' num2str(numLastVar)],['x' num2str(numLastVar + 1)]);
numLastVar = numLastVar + 1;


addLayer(obj.net,[finalFCName 'Prt'], dagnn.Conv('size',[1 1 4096 numPrtClasses]),...
    ['x' num2str(numLastVar)],['x' num2str(numLastVar +1)],...
    {[finalFCName 'Prtf'],[finalFCName 'Prtb']});

% Initialize parameters
finalFCIdx = obj.net.getLayerIndex([finalFCName 'Prt']);
newParams = obj.net.layers(finalFCIdx).block.initParams();

% Initialize parameters
obj.net.params(obj.net.layers(finalFCIdx).paramIndexes(1)).value = newParams{1} / std(newParams{1}(:)) * 0.001; % Girshick initialization with std of 0.001
obj.net.params(obj.net.layers(finalFCIdx).paramIndexes(2)).value = newParams{2};


%%% Add losses
softmaxlossBlock = dagnn.LossWeighted('loss', 'softmaxlog');
addLayer(obj.net, 'softmaxlossObj', softmaxlossBlock,...
    {obj.net.layers(obj.net.getLayerIndex([finalFCName 'Obj'])).outputs{1}, 'labelObj','instanceWeightsObj'},{'objectiveObj'});


softmaxlossBlock = dagnn.LossWeighted('loss', 'softmaxlog');
addLayer(obj.net, 'softmaxlossPrt', softmaxlossBlock,...
    {obj.net.layers(obj.net.getLayerIndex([finalFCName 'Prt'])).outputs{1}, 'labelPrt','instanceWeightsPrt'},{'objectivePrt'});



%%% Add bounding box regression layer
if obj.nnOpts.bboxRegress
    finalFCLayerIdx = obj.net.getLayerIndex([finalFCName 'Obj']);
    inputVars = obj.net.layers(finalFCLayerIdx).inputs;
    finalFCLayerSize = size(obj.net.params(obj.net.layers(finalFCLayerIdx).paramIndexes(1)).value);
    regressLayerSize = finalFCLayerSize .* [1 1 1 4]; % Four times bigger than classification layer
    regressName = [finalFCName 'regressObj'];
    obj.net.addLayer(regressName, dagnn.Conv('size', regressLayerSize), inputVars, {'regressionScoreObj'}, {'regressObjf', 'regressObjb'});
    regressIdx = obj.net.getLayerIndex(regressName);
    newParams = obj.net.layers(regressIdx).block.initParams();
    obj.net.params(obj.net.layers(regressIdx).paramIndexes(1)).value = newParams{1} / std(newParams{1}(:)) * 0.001; % Girshick initialization with std of 0.001
    obj.net.params(obj.net.layers(regressIdx).paramIndexes(2)).value = newParams{2};

    obj.net.addLayer('regressLossObj', dagnn.LossRegress('loss', 'Smooth', 'smoothMaxDiff', 1), ...
        {'regressionScoreObj', 'regressionTargetsObj', 'instanceWeightsObj'}, 'regressObjectiveObj');
    
    
    finalFCLayerIdx = obj.net.getLayerIndex([finalFCName 'Prt']);
    inputVars = obj.net.layers(finalFCLayerIdx).inputs;
    finalFCLayerSize = size(obj.net.params(obj.net.layers(finalFCLayerIdx).paramIndexes(1)).value);
    regressLayerSize = finalFCLayerSize .* [1 1 1 4]; % Four times bigger than classification layer
    regressName = [finalFCName 'regressPrt'];
    obj.net.addLayer(regressName, dagnn.Conv('size', regressLayerSize), inputVars, {'regressionScorePrt'}, {'regressPrtf', 'regressPrtb'});
    regressIdx = obj.net.getLayerIndex(regressName);
    newParams = obj.net.layers(regressIdx).block.initParams();
    obj.net.params(obj.net.layers(regressIdx).paramIndexes(1)).value = newParams{1} / std(newParams{1}(:)) * 0.001; % Girshick initialization with std of 0.001
    obj.net.params(obj.net.layers(regressIdx).paramIndexes(2)).value = newParams{2};

    obj.net.addLayer('regressLossPrt', dagnn.LossRegress('loss', 'Smooth', 'smoothMaxDiff', 1), ...
        {'regressionScorePrt', 'regressionTargetsPrt', 'instanceWeightsPrt'}, 'regressObjectivePrt');
    
end

%%% Set correct learning rates and biases (Girshick style)
if obj.nnOpts.fastRcnnParams
    % Biases have learning rate of 2 and no weight decay
    for lI = 1 : length(obj.net.layers)
        if isa(obj.net.layers(lI).block, 'dagnn.Conv') % ADD SOMETHING HERE FOR DIFFERENT CONVs
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
