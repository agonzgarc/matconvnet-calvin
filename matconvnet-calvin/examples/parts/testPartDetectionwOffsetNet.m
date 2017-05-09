function [results] = testPartDetectionwOffsetNet(imdb, nnOpts, net, inputs, ~)
% [results] = testDetection(imdb, nnOpts, net, inputs, ~)
%
% Get predicted boxes and scores per class
% Only gets top nnOpts.maxNumBoxesPerImTest boxes (default: 5000)
% Only gets boxes with score higher than nnOpts.minDetectionScore (default: 0.01)
% NMS threshold: nnOpts.nmsTTest (default: 0.3)

% Variables which should probably be in imdb.nnOpts or something
% Jasper: Probably need to do something more robust here
%
% Copyright by Jasper Uijlings, 2015
% Modified by Abel Gonzalez-Garcia, 2016

if isfield(nnOpts, 'maxNumBoxesPerImTest')
    maxNumBoxesPerImTest = nnOpts.maxNumBoxesPerImTest;
else
    maxNumBoxesPerImTest = 5000;
end

if isfield(nnOpts, 'nmsTTest')
    nmsTTest = imdb.nmsTTest;
else
    nmsTTest = 0.3; % non-maximum threshold
end

if isfield(nnOpts, 'minDetectionScore')
    minDetectionScore = nnOpts.minDetectionScore;
else
    minDetectionScore = 0.01;
end


%% Parts

% Get scores
vI = net.getVarIndex('scoresPrt');
scoresStruct = net.vars(vI);
scores = permute(scoresStruct.value, [4 3 2 1]);

% Get boxes
inputNames = inputs(1:2:end);
[~, boxI] = ismember('boxesPrt', inputNames);
boxI = boxI * 2; % Index of actual argument
boxes = inputs{boxI}';

% Get regression targets for boxes
if imdb.boxRegress
    vI = net.getVarIndex('regressionScorePrt');
    regressStruct = net.vars(vI);
    regressFactors = permute(regressStruct.value, [4 3 2 1]);
else
    regressFactors = zeros(size(boxes,1), size(boxes,2) * imdb.numClassesPrt);
end


% Get top boxes for each category but do NOT perform NMS.
currMaxBoxes = min(maxNumBoxesPerImTest, size(boxes, 1));

for cI = size(scores,2) : -1 : 1
    % Get top scores and boxes
    [currScoresT, sI] = sort(scores(:,cI), 'descend');
    currScoresT = currScoresT(1:currMaxBoxes);
    sI = sI(1:currMaxBoxes);
    currBoxes = boxes(sI,:);
    
    % Do regression
    regressFRange = (cI*4)-3:cI*4;
    currRegressF = gather(regressFactors(sI,regressFRange));
    currBoxesReg = BoxRegresssGirshick(currBoxes, currRegressF);
        
    
    % Get scores (w boxes) above certain threshold
    % In theory, we should keep all boxes as they might surpass minDetScore
    % when combined with the other scores. However, this would be demanding
    % in terms of storage, and when the situation arises it will be barely
    % noticeable
    goodI = currScoresT > minDetectionScore;
    currScoresT = currScoresT(goodI, :);
    currBoxes = currBoxes(goodI, :);

    results.boxesPrt{cI} = gather(currBoxes);
    results.scoresPrt{cI} = gather(currScoresT);
   
    if imdb.boxRegress
        currBoxesReg = currBoxesReg(goodI, :);
        results.boxesRegressedPrt{cI} = gather(currBoxesReg);
        % No need to store scores for the regressed boxes --> same as
        % non-regressed as we haven't performed NMS
    end
end


%% Objects

% Get scores
vI = net.getVarIndex('scoresObj');
scoresStruct = net.vars(vI);
scores = permute(scoresStruct.value, [4 3 2 1]);

% Get boxes
inputNames = inputs(1:2:end);
[~, boxI] = ismember('boxesObj', inputNames);
boxI = boxI * 2; % Index of actual argument
boxes = inputs{boxI}';


% Get regression targets for boxes
if imdb.boxRegress
    vI = net.getVarIndex('regressionScoreObj');
    regressStruct = net.vars(vI);
    regressFactors = permute(regressStruct.value, [4 3 2 1]);
else
    regressFactors = zeros(size(boxes,1), size(boxes,2) * imdb.numClasses);
end

% Get top boxes for each category. Perform NMS. Thresholds defined at top of function
currMaxBoxes = min(maxNumBoxesPerImTest, size(boxes, 1));

% Do not save info for background class (1)
for cI = size(scores,2) : -1 :2
    % Get top scores and boxes
    [currScoresT, sI] = sort(scores(:,cI), 'descend');
    currScoresT = currScoresT(1:currMaxBoxes);
    sI = sI(1:currMaxBoxes);
    currBoxes = boxes(sI,:);

    % Do regression
    regressFRange = (cI*4)-3:cI*4;
    currRegressF = gather(regressFactors(sI,regressFRange));
    currBoxesReg = BoxRegresssGirshick(currBoxes, currRegressF);
    
    % Get scores (w boxes) above certain threshold
    goodI = currScoresT > minDetectionScore;
    currScoresT = currScoresT(goodI, :);
    currBoxes = currBoxes(goodI, :);
    currBoxesReg = currBoxesReg(goodI, :);

    % Perform NMS
    [~, goodBoxesI] = BoxNMS(currBoxes, nmsTTest);
    currBoxes = currBoxes(goodBoxesI, :);
    currScores = currScoresT(goodBoxesI ,:);
    
    results.boxesObj{cI} = gather(currBoxes);
    results.scoresObj{cI} = gather(currScores);
    
    if imdb.boxRegress
        [~, goodBoxesI] = BoxNMS(currBoxesReg, nmsTTest);
        currBoxesReg = currBoxesReg(goodBoxesI, :);
        currScoresRegressed = currScoresT(goodBoxesI, :);
        results.boxesRegressedObj{cI} = gather(currBoxesReg);
        results.scoresRegressedObj{cI} = gather(currScoresRegressed);
    end
end

%% Rel Loc

regressFactors = cell(max(nnOpts.misc.numOuts),1);

% Get displaced windows regression scores
for idxNum = 1:4
    vI = net.getVarIndex(['dispWindows' num2str(idxNum) 'Scores']);
    regressStruct = net.vars(vI);
    regressFactors{idxNum} = permute(regressStruct.value, [4 3 2 1]);
end

vI = net.getVarIndex('objDets');
objDets = squeeze(net.vars(vI).value)';

vI = net.getVarIndex('objDetsScores');
objDetsScores = squeeze(net.vars(vI).value)';

vI = net.getVarIndex('presenceScores');
presenceScores = squeeze(net.vars(vI).value);

% Create presence map to get which displaced windows belong to part class
mapPartPresence = zeros(sum(nnOpts.misc.numOuts),1);
kk = 1;
for idxPart = 1:105
    mapPartPresence(kk:kk+nnOpts.misc.numOuts(idxPart)-1) = idxPart;
    kk = kk + nnOpts.misc.numOuts(idxPart);
end


for cI = size(regressFactors{idxNum},2)/4 : -1 : 1
    
    % Get object detections for class of object
    idxStart = (nnOpts.misc.idxPartGlobal2idxClass(cI)-1)*5+1;

    % Always 5 detections, although some might be dummy
    objDetsClass = objDets(idxStart:idxStart+4,:);
    objDetsScoresClass = objDetsScores(idxStart:idxStart+4);
    presenceScoresClass = 1./(1+ exp(-presenceScores(mapPartPresence == cI,idxStart:idxStart+4)'));
    
    % Remove info from dummy detections
    idxDummy = objDetsScoresClass == -1;
    
    numObjDets = sum(~idxDummy);
    
    objDetsClass = objDetsClass(~idxDummy,:);
    objDetsScoresClass = objDetsScoresClass(~idxDummy,:);
    presenceScoresClass = presenceScoresClass(~idxDummy,:);

    
    regressFRange = (cI*4)-3:cI*4;
    
    dispWindows = zeros(gather(numObjDets*nnOpts.misc.numOuts(cI)),4);
    kk = 1;
    for idxNum = 1:nnOpts.misc.numOuts(cI)
        % Regress factor for mode-class combination
        currRegressF = gather(regressFactors{idxNum}(idxStart:idxStart+4,regressFRange));
        dispWindows(kk:kk+numObjDets-1,:) =  BoxRegresssGirshick(objDetsClass, currRegressF(~idxDummy,:));
        kk = kk + numObjDets;
    end
    
    results.dispWindows{cI} = single(gather(dispWindows));
    results.presenceScores{cI} = gather(reshape(presenceScoresClass,numel(presenceScoresClass),1));
    results.objDets{cI} = gather(objDetsClass);
    results.objDetsScores{cI} = gather(objDetsScoresClass);

end



