function mergeObjAppClswOffsetNet(obj, varargin)
%
% Combine Obj. App+Cls network with OffsetNet, given as an argument
%
% Created by Abel Gonzalez-Garcia, 2016

% Initial settings
p = inputParser;
addParameter(p, 'numOuts','4');
addParameter(p, 'offsetNet','0');
parse(p, varargin{:});


maxNumOuts = p.Results.numOuts;
offsetNet = p.Results.offsetNet;



%% Copy over all offsetNet layers

% Define input names that should stay the same
sharedInputs = {'input', 'oriImSize', 'instanceWeights'};

% Define Offset net  outputs that are also inputs (already in new net)
fixedOutputs = {'presenceScores'};

for ii = 1:maxNumOuts
   fixedOutputs{end+1} = ['dispWindows' num2str(ii) 'Scores']; %#ok<AGROW>
end

% Go backwards so there's less confussion with names
for idxLayer = size(offsetNet.layers,2):-1:1
    
    % Add ON suffix to differentiate form other network, only if name
    % already exists
    name = offsetNet.layers(idxLayer).name;
    if ~isnan(getLayerIndex(obj.net,name))
        name = [name 'ON']; %#ok<AGROW>
    end
    block = offsetNet.layers(idxLayer).block;
    inputs = offsetNet.layers(idxLayer).inputs;
    for kk = 1:numel(inputs)
        if sum(strcmp(inputs{kk},sharedInputs)) == 0 && ...
                ~isnan(getVarIndex(obj.net,inputs{kk}))
            inputs{kk} = [inputs{kk} 'ON'];
        end
    end
    outputs = offsetNet.layers(idxLayer).outputs;
    for kk = 1:numel(outputs)
        if sum(strcmp(outputs{kk},fixedOutputs)) == 0 && ...
                ~isnan(getVarIndex(obj.net,outputs{kk}))
           outputs{kk} = [ outputs{kk} 'ON'];
        end
    end
    
    params = offsetNet.layers(idxLayer).params;
    for kk = 1:numel(params)
        if ~isnan(getParamIndex(obj.net,params{kk}))
           params{kk} = [ params{kk} 'ON'];
        end
    end
    
    addLayer(obj.net, name, block, inputs, outputs, params);
    
    % Copy parameters
    paramIdxON = offsetNet.layers(idxLayer).paramIndexes;
    paramIdxNewNet = obj.net.layers(getLayerIndex(obj.net,name)).paramIndexes;

    for idxParam = 1:numel(paramIdxON)
        obj.net.params(paramIdxNewNet(idxParam)).value = ...
            offsetNet.params(paramIdxON(idxParam)).value;
    end
end


%% Do adaptation to test here (maybe leave some for later)

% Replace softmaxloss layer with softmax layer
softMaxLossIdx = obj.net.getLayerIndex('softmaxlossObj');
if ~isnan(softMaxLossIdx)
    softmaxlossInput = obj.net.layers(softMaxLossIdx).inputs{1};
    obj.net.removeLayer('softmaxlossObj');
    obj.net.addLayer('softmaxObj', dagnn.SoftMax(), softmaxlossInput, 'scoresObj', {});
    softmaxIdx = obj.net.layers(obj.net.getLayerIndex('softmaxObj')).outputIndexes;
    assert(numel(softmaxIdx) == 1);
end

softMaxLossIdx = obj.net.getLayerIndex('softmaxlossPrt');
if ~isnan(softMaxLossIdx)
    softmaxlossInput = obj.net.layers(softMaxLossIdx).inputs{1};
    obj.net.removeLayer('softmaxlossPrt');
    obj.net.addLayer('softmaxPrt', dagnn.SoftMax(), softmaxlossInput, 'scoresPrt', {});
    softmaxIdx = obj.net.layers(obj.net.getLayerIndex('softmaxPrt')).outputIndexes;
    assert(numel(softmaxIdx) == 1);
end

% Remove regression loss if it's there
regressLossIdx = obj.net.getLayerIndex('regressLossPrt');
if ~isnan(regressLossIdx)
   obj.net.removeLayer('regressLossPrt'); 
end

% Remove regression loss if it's there
regressLossIdx = obj.net.getLayerIndex('regressLossObj');
if ~isnan(regressLossIdx)
   obj.net.removeLayer('regressLossObj'); 
end

% Replace softmaxloss layer with softmax layer
for idxOut = 1:4
    dWLossIdx = obj.net.getLayerIndex(['dispWindows' num2str(idxOut) 'Loss']);
    if ~isnan(dWLossIdx)
        obj.net.removeLayer(['dispWindows' num2str(idxOut) 'Loss']);
    end
end

presenceLossIdx = obj.net.getLayerIndex('presenceLogLoss');
if ~isnan(presenceLossIdx)
    obj.net.removeLayer('presenceLogLoss');
end


% Add new layer to process detections
% Its input is the first two inputs of suppSel
suppSelIdx = obj.net.getLayerIndex('suppSel');
inputVars = obj.net.layers(suppSelIdx).inputs;

addLayer(obj.net, 'objDetsRelLoc', dagnn.ObjDetections(),inputVars(1:2),...
    {'objDets','objDetsScores'},{});

% Keep object detections for later regression3
vI = obj.net.getVarIndex('objDets');
obj.net.vars(vI).precious = true;


sortLayers(obj.net);

