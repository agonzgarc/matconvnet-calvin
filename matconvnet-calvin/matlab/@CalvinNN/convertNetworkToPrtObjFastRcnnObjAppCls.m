function convertNetworkToPrtObjFastRcnnObjAppCls(obj, varargin)
%
% Modify network for Fast R-CNN's ROI pooling.
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
parse(p, varargin{:});

lastConvPoolName = p.Results.lastConvPoolName;
firstFCName = p.Results.firstFCName;
secondFCName =  p.Results.secondFCName;
finalFCName = p.Results.finalFCName;
numObjClasses = p.Results.numObjClasses;
numPrtClasses = p.Results.numPrtClasses;


% Get number of last variable
idxLastLayer = obj.net.getLayerIndex([finalFCName 'Prt']);
numLastVar = str2double(obj.net.layers(idxLastLayer).outputs{1}(2:end));


% We need a softmax for the object scores now (for supporting box select)
addLayer(obj.net, 'softmaxSelObj', dagnn.SoftMax(), obj.net.layers(obj.net.getLayerIndex([finalFCName 'Obj'])).outputs{1}, ...
    ['x' num2str(numLastVar + 1)]);
numLastVar = numLastVar + 1;

addLayer(obj.net, 'suppSel', dagnn.SupportingSelector(), {['x' num2str(numLastVar)],'boxesObj', 'boxesPrt', 'insideness'},...
    {'suppObjScores', 'suppObjBox','idxSupp'}); 


% Add external fc layers for object appearance with its own RoiPool
lastConvPoolIdx = obj.net.getLayerIndex(['roi', lastConvPoolName 'Obj']);

firstFCIdx = obj.net.layers(lastConvPoolIdx).outputIndexes;
roiPoolSize = obj.net.layers(firstFCIdx(2)).block.size(1:2);

roiPoolName = ['roi', lastConvPoolName 'ObjApp'];
roiPoolBlock = dagnn.RoiPooling('poolSize', roiPoolSize);
addLayer(obj.net, roiPoolName, roiPoolBlock,...
    {obj.net.layers(lastConvPoolIdx).inputs{1},'oriImSize', 'suppObjBox'},...
    {['x' num2str(numLastVar +1)], 'roiPoolMaskObjApp'});
numLastVar = numLastVar + 1;

addLayer(obj.net, [firstFCName 'ObjApp'],...
    dagnn.Conv('size',obj.net.layers(obj.net.getLayerIndex([firstFCName 'Obj'])).block.size),...
    ['x' num2str(numLastVar)], ['x' num2str(numLastVar+1)], {[firstFCName 'ObjAppf'],[firstFCName 'ObjAppb']});
numLastVar = numLastVar + 1;

% Init params with the pre-trained network params (now in obj branch)   
idxParamsObj = obj.net.layers(obj.net.getLayerIndex([firstFCName 'Obj'])).paramIndexes;
idxParamsObjApp = obj.net.layers(obj.net.getLayerIndex([firstFCName 'ObjApp'])).paramIndexes;
obj.net.params(idxParamsObjApp(1)).value = obj.net.params(idxParamsObj(1)).value;
obj.net.params(idxParamsObjApp(2)).value = obj.net.params(idxParamsObj(2)).value;

addLayer(obj.net, ['relu' firstFCName(end) 'ObjApp'], dagnn.ReLU,...
    ['x' num2str(numLastVar)], ['x' num2str(numLastVar+1)],{});
numLastVar = numLastVar + 1;

addLayer(obj.net, ['dropout' firstFCName(end) 'ObjApp'],dagnn.DropOut(),...
    ['x' num2str(numLastVar)],['x' num2str(numLastVar + 1)]);
numLastVar = numLastVar + 1;

addLayer(obj.net, [secondFCName 'ObjApp'],...
    dagnn.Conv('size',obj.net.layers(obj.net.getLayerIndex([secondFCName 'Obj'])).block.size),...
    ['x' num2str(numLastVar)], ['x' num2str(numLastVar+1)], {[secondFCName 'ObjAppf'],[secondFCName 'ObjAppb']});
numLastVar = numLastVar + 1;

% Init params with the pre-trained network params (now in obj branch)   
idxParamsObj = obj.net.layers(obj.net.getLayerIndex([secondFCName 'Obj'])).paramIndexes;
idxParamsObjApp = obj.net.layers(obj.net.getLayerIndex([secondFCName 'ObjApp'])).paramIndexes;
obj.net.params(idxParamsObjApp(1)).value = obj.net.params(idxParamsObj(1)).value;
obj.net.params(idxParamsObjApp(2)).value = obj.net.params(idxParamsObj(2)).value;

addLayer(obj.net, ['relu' secondFCName(end) 'ObjApp'], dagnn.ReLU,...
    ['x' num2str(numLastVar)], ['x' num2str(numLastVar+1)],{});
numLastVar = numLastVar + 1;

addLayer(obj.net, ['dropout' secondFCName(end) 'ObjApp'],dagnn.DropOut(),...
    ['x' num2str(numLastVar)],['x' num2str(numLastVar + 1)]);
numLastVar = numLastVar + 1;



% And now add combination 
replaceLayer(obj.net, [finalFCName 'Prt'], [finalFCName 'PrtwObjAppCls'],...
    dagnn.ConvCombination('size',[1 1 2*4096+numObjClasses numPrtClasses],'numInputs',4),...
    {'suppObjScores', ['x' num2str(numLastVar)], 'idxSupp'});

% Initialize parameters
finalFCIdx = obj.net.getLayerIndex([finalFCName 'PrtwObjAppCls']);
newParams = obj.net.layers(finalFCIdx).block.initParams();

% Initialize parameters
obj.net.params(obj.net.layers(finalFCIdx).paramIndexes(1)).value = newParams{1} / std(newParams{1}(:)) * 0.001; % Girshick initialization with std of 0.001
obj.net.params(obj.net.layers(finalFCIdx).paramIndexes(2)).value = newParams{2};

%%% Add bounding box regression layer
if obj.nnOpts.bboxRegress
 
    finalFCLayerIdx = obj.net.getLayerIndex([finalFCName 'PrtwObjAppCls']);
    inputVars = obj.net.layers(finalFCLayerIdx).inputs;
    finalFCLayerSize = size(obj.net.params(obj.net.layers(finalFCLayerIdx).paramIndexes(1)).value);
    regressLayerSize = finalFCLayerSize .* [1 1 1 4]; % Four times bigger than classification layer
    regressName = [finalFCName 'regressPrt'];
    replaceLayer(obj.net,regressName, [regressName 'wObjAppCls'], dagnn.ConvCombination('size', regressLayerSize,'numInputs',4), inputVars(2:end));
    regressIdx = obj.net.getLayerIndex([regressName 'wObjAppCls']);
    newParams = obj.net.layers(regressIdx).block.initParams();
    obj.net.params(obj.net.layers(regressIdx).paramIndexes(1)).value = newParams{1} / std(newParams{1}(:)) * 0.001; % Girshick initialization with std of 0.001
    obj.net.params(obj.net.layers(regressIdx).paramIndexes(2)).value = newParams{2};
    
end

sortLayers(obj.net);

%%% Set correct learning rates and biases (Girshick style)
if obj.nnOpts.fastRcnnParams
    % Biases have learning rate of 2 and no weight decay
    for lI = 1 : length(obj.net.layers)
        if isa(obj.net.layers(lI).block, 'dagnn.Conv') || isa(obj.net.layers(lI).block, 'dagnn.ConvCombination')
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
