function[stats] = testPrtObj(obj)
% [stats] = test(obj)
%
% Test function
% - Does a single processing of an epoch for testing
% - Uses the nnOpts.testFn function for the testing (inside process_epoch)
% - Automatically changes softmaxloss to softmax, removes hinge loss. Other losses are not yet supported
%
% Copyright by Jasper Uijlings, 2015
% Modified by Holger Caesar, 2016
% 
% Modified by Abel Gonzalez-Garcia, 2016

% Check that we only use one GPU
numGpus = numel(obj.nnOpts.gpus);
assert(numGpus <= 1);
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

% Set datasetMode in imdb
datasetMode = 'test';
obj.net.mode = datasetMode; % Disable dropout
obj.imdb.setDatasetMode(datasetMode);
state.epoch = 1;
state.allBatchInds = obj.imdb.getAllBatchInds();

% Process the epoch
obj.stats.(datasetMode) = obj.processEpoch(obj.net, state);

% The stats are the desired results
stats = obj.stats.(datasetMode);