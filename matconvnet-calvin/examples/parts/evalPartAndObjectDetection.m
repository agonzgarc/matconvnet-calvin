function evalPartAndObjectDetection(testName, stats, nnOpts)

global DATAopts;

% Get test images
trash = load(sprintf(DATAopts.imdb, testName));
imdbTest = trash.imdb;
clear trash;
% Use images with at least one part
testIms = imdbTest.image_ids(unique(imdbTest.mapping(:,4)));

% get image sizes
testCount = length(testIms);
imSizes = imdbTest.sizes(unique(imdbTest.mapping(:,4)),:);

% Add imdb to DATAopts for part detection evaluation
DATAopts.imdbTest = imdbTest;

[DATAopts.prt_classes, idxPartGlobal2idxClass]  = getPartNames(imdbTest);

%% Parts

for cI = 1:105    
    %%
    currBoxes = cell(length(testIms), 1);
    currScores = cell(length(testIms), 1);
    for i=1:length(testIms)
        currBoxes{i} = stats.results(i).boxesPrt{cI+1};
        currScores{i} = stats.results(i).scoresPrt{cI+1};
    end
    
    [currBoxes, fileIdx] = Cell2Matrix(gather(currBoxes));
    [currScores, fileIdx2] = Cell2Matrix(gather(currScores));
    
    assert(isequal(fileIdx, fileIdx2)); % Should be equal

    currFilenames = testIms(fileIdx);
    
    [~, sI] = sort(currScores, 'descend');
    currScores = currScores(sI);
    currBoxes = currBoxes(sI,:);
    currFilenames = currFilenames(sI);
  
    [recallPrt{cI}, precPrt{cI}, apPrt(cI,1), upperBoundPrt{cI}] = ...
            DetectionPartsToPascalVOCFiles(testName, cI, idxPartGlobal2idxClass(cI+1) , currBoxes, currFilenames, currScores, ...
                                           'Matconvnet-Calvin-Prt', 1, 0);
        apPrt(cI)
end

if isfield(stats.results(1), 'boxesRegressedPrt')
    for cI = 1:105
        %%
        currBoxes = cell(length(testIms), 1);
        currScores = cell(length(testIms), 1);
        for i=1:length(testIms)
            % Get regressed boxes and refit them to the image
            currBoxes{i} = stats.results(i).boxesRegressedPrt{cI+1};
            currBoxes{i}(:,1) = max(currBoxes{i}(:,1), 1);
            currBoxes{i}(:,2) = max(currBoxes{i}(:,2), 1);
            currBoxes{i}(:,3) = min(currBoxes{i}(:,3), imSizes(i,2));
            currBoxes{i}(:,4) = min(currBoxes{i}(:,4), imSizes(i,1));

            currScores{i} = stats.results(i).scoresRegressedPrt{cI+1};
        end

        [currBoxes, fileIdx] = Cell2Matrix(gather(currBoxes));
        [currScores, fileIdx2] = Cell2Matrix(gather(currScores));

        isequal(fileIdx, fileIdx2) % Should be equal

        currFilenames = testIms(fileIdx);

        [~, sI] = sort(currScores, 'descend');
        currScores = currScores(sI);
        currBoxes = currBoxes(sI,:);
        currFilenames = currFilenames(sI);

        %%
         [recallPrt{cI}, precPrt{cI}, apRegressedPrt(cI,1), upperBoundPrt{cI}] = ...
            DetectionPartsToPascalVOCFiles(testName, cI, idxPartGlobal2idxClass(cI+1) , currBoxes, currFilenames, currScores, ...
                                           'Matconvnet-Calvin-Prt', 1, 0);
        apRegressedPrt(cI)
    end
    
    apRegressedPrt
    mean(apRegressedPrt)
    
else
     apRegressedPrt = 0;
end


for cI = 1 : 20
    %
    currBoxes = cell(testCount, 1);
    currScores = cell(testCount, 1);
    for i = 1 : testCount
        currBoxes{i} = stats.results(i).boxesObj{cI + 1};
        currScores{i} = stats.results(i).scoresObj{cI + 1};
    end
    
    [currBoxes, fileIdx] = Cell2Matrix(gather(currBoxes));
    [currScores, fileIdx2] = Cell2Matrix(gather(currScores));
    
    assert(isequal(fileIdx, fileIdx2)); % Should be equal
    
    currFilenames = testIms(fileIdx);
    
    [~, sI] = sort(currScores, 'descend');
    currScores = currScores(sI);
    currBoxes = currBoxes(sI,:);
    currFilenames = currFilenames(sI);

    [recallObj{cI}, precObj{cI}, apObj(cI,1), upperBoundObj{cI}] = ...
        DetectionToPascalVOCFiles(testName, cI, currBoxes, currFilenames, currScores, ...
        'Matconvnet-Calvin-Obj', 1, nnOpts.misc.overlapNms);
    apObj(cI)
end

apObj
mean(apObj)

if isfield(stats.results(1), 'boxesRegressedObj')
    for cI = 1 : 20
        %
        currBoxes = cell(testCount, 1);
        currScores = cell(testCount, 1);
        
        for i=1:testCount
            % Get regressed boxes and refit them to the image
            currBoxes{i} = stats.results(i).boxesRegressedObj{cI+1};
            currBoxes{i}(:,1) = max(currBoxes{i}(:, 1), 1);
            currBoxes{i}(:,2) = max(currBoxes{i}(:, 2), 1);
            currBoxes{i}(:,3) = min(currBoxes{i}(:, 3), imSizes(i,2));
            currBoxes{i}(:,4) = min(currBoxes{i}(:, 4), imSizes(i,1));
            
            currScores{i} = stats.results(i).scoresRegressedObj{cI+1};
        end
        
        [currBoxes, fileIdx] = Cell2Matrix(gather(currBoxes));
        [currScores, fileIdx2] = Cell2Matrix(gather(currScores));
        
        assert(isequal(fileIdx, fileIdx2)); % Should be equal
        
        currFilenames = testIms(fileIdx);
        
        [~, sI] = sort(currScores, 'descend');
        currScores = currScores(sI);
        currBoxes = currBoxes(sI, :);
        currFilenames = currFilenames(sI);
        
        %     ShowImageRects(currBoxes(1:32, [2 1 4 3]), 4, 4, currFilenames(1:32), currScores(1:32));
        
        %
        [recallObj{cI}, precObj{cI}, apRegressedObj(cI,1), upperBoundObj{cI}] = ...
            DetectionToPascalVOCFiles(testName, cI, currBoxes, currFilenames, currScores, ...
            'Matconvnet-Calvin-Obj', 1, nnOpts.misc.overlapNms);
        apRegressedObj(cI)
    end
    
    apRegressedObj
    mean(apRegressedObj)
else
    apRegressedObj = 0;
end

% Save results to disk
save([nnOpts.expDir, '/', 'resultsEpochFinalTest.mat'], 'nnOpts', 'stats', 'apPrt', 'apRegressedPrt', 'apObj', 'apRegressedObj');