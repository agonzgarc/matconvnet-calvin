function CM = trainCoefficients(testName, stats)

% Training of coefficients for mixing with OffsetNet

global DATAopts;

% Get test images
trash = load(sprintf(DATAopts.imdb, testName));
imdbTest = trash.imdb;
clear trash;
% Use images with at least one part
testIms = imdbTest.image_ids(unique(imdbTest.mapping(:,4)));

% get image sizes
testCount = length(testIms);

coeffs = 0:0.1:1;

ap = zeros(105, numel(coeffs));

% Add imdb to DATAopts for part detection evaluation
DATAopts.imdbTest = imdbTest;

[DATAopts.prt_classes, idxPartGlobal2idxClass]  = getPartNames(imdbTest);

minDetectionScore = 0.01;


for kk = 1:numel(coeffs)

    for cI =1:105   
        %%
        currBoxes = cell(length(testIms), 1);
        currScores = cell(length(testIms), 1);

        for i=1:testCount

            currNetBoxes = stats.results(i).boxesPrt{cI+1};
            currNetScores = stats.results(i).scoresPrt{cI+1};
            
            % Assume stats has ON info if nnOpts has coefficients field
            scoresObjDets = stats.results(i).objDetsScores{cI};
            dispWindows = stats.results(i).dispWindows{cI};
            presenceScores = stats.results(i).presenceScores{cI};
            
             % Only consider those displaced windows the ones that 
            windowWeights = repmat(scoresObjDets,numel(presenceScores)...
                /numel(scoresObjDets),1).*presenceScores;
            
            scoreFromDispWindows  = scoreBoxesWithDispWindowsUnsorted(dispWindows, currNetBoxes, windowWeights);

            newScoresNet = (1-coeffs(cI))*currNetScores + ...
                coeffs(cI)*scoreFromDispWindows;

            boxes = currNetBoxes;
            scores = newScoresNet;
            
            % Sort, get detections > thresh and perform NMS
            [currNetScoresT, sI] = sort(scores, 'descend');
            currNetBoxesT = boxes(sI,:);

            goodI = currNetScoresT > minDetectionScore;
            currScoresT = currNetScoresT(goodI, :);
            currBoxesT= currNetBoxesT(goodI, :);

            [~, goodBoxesI] = BoxNMS(currBoxesT);
            currBoxes{i} = currBoxesT(goodBoxesI, :);
            currScores{i} = currScoresT(goodBoxesI);
        end

        [currBoxes, fileIdx] = Cell2Matrix(gather(currBoxes));
        [currScores, ~] = Cell2Matrix(gather(currScores));

        currFilenames = testIms(fileIdx);

        [~, sI] = sort(currScores, 'descend');
        currScores = currScores(sI);
        currBoxes = currBoxes(sI,:);
        currFilenames = currFilenames(sI);

        [~, ~, ap(cI,kk), ~] = ...
                DetectionPartsToPascalVOCFiles(testName, cI, idxPartGlobal2idxClass(cI+1) , currBoxes, currFilenames, currScores, ...
                                               'Matconvnet-Calvin-Prt-Coeff', 1, 0);
    end
end


[~,idxMax] = max(ap,[],2);


CM = coeffs(idxMax)';
