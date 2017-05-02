function[imdb] = setupImdbPartDetection(imdbFunc,trainName, testName, net, ONparams)

global DATAopts;

%%% Setup the Imdb
% Get and test images from imbd
trash = load(sprintf(DATAopts.imdb, trainName));
imdbTrain = trash.imdb;
trash = load(sprintf(DATAopts.imdb, testName));
imdbTest = trash.imdb;
clear trash;
% Consider only images with at least one part 
trainIms = imdbTrain.image_ids;
testIms = imdbTest.image_ids;

if nargin < 5
    ONparams = [];
end

% Make train, val, and test set. For Pascal, I illegally use part of the test images
% as validation set. This is to match Girshick performance while still having
% meaningful graphs for the validation set.
% Note: allIms are just all images. datasetIdx determines how these are divided over
% train, val, and test.
allIms = cat(1, trainIms, testIms);
datasetIdx = cell(3, 1);
datasetIdx{1} = (1:length(trainIms))';  % Jasper: Use all training images. Only for comparison Pascal Girshick
datasetIdx{2} = (length(trainIms)+1:length(trainIms)+501)'; % Use part of the test images for validation. Not entirely legal, but otherwise it will take much longer to get where we want.
datasetIdx{3} = (length(trainIms)+1:length(allIms))';

if isempty(ONparams)
    imdb = imdbFunc(DATAopts.imgpath(1:end-6), ...        % path
        DATAopts.imgpath(end-3:end), ...      % image extension
        DATAopts.gStructPath, ...             % gStruct path
        allIms, ...                           % all images
        datasetIdx, ...                       % division into train/val/test
        net.meta.normalization.averageImage);      % average image used to pretrain network
else
    % Get mapping from global part idx to class idx
    [~, idxPartGlobal2idxClass]  = getPartNames(imdbTrain);
    
    imdb = imdbFunc(DATAopts.imgpath(1:end-6), ...        % path
        DATAopts.imgpath(end-3:end), ...      % image extension
        DATAopts.gStructPath, ...             % gStruct path
        allIms, ...                           % all images
        datasetIdx, ...                       % division into train/val/test
        net.meta.normalization.averageImage,...  % average image used to pretrain network
        ONparams.numOuts,...            % extra Offset Net params
        idxPartGlobal2idxClass,...
        ONparams.idxObjClassRem);      
end

% Usually instance weighting gives better performance. But not Girshick style
% imdbPascal.SetInstanceWeighting(true);

% Store lists for use in eval
imdb.misc.trainIms  = trainIms;
imdb.misc.testIms   = testIms;