function setupParts(varargin)
% setupParts(varargin)
%
% This includes extracting Selective Search proposals and ground-truth for 
% each image in the PASCAL VOC 20xx dataset. Note that this takes about 
% 4s/im or about 11h for VOC 2010.
%
% Copyright by Holger Caesar, 2016

%%% Settings
% Dataset
vocYear = 2010;
trainName = 'train';
testName  = 'val';
vocName = sprintf('VOC%d', vocYear);
global glDatasetFolder;
datasetDir = [fullfile(glDatasetFolder, vocName), '/'];
setupDataOptsPrts(vocYear, testName, datasetDir);
global DATAopts; % Database specific paths
assert(~isempty(DATAopts), 'Error: Dataset not initialized properly!');

%% Create IMDBs
imSet = 'train';
imdb = createIMDB(imSet);
save(sprintf(DATAopts.imdb, imSet), 'imdb');


imSet = 'val';
imdb = createIMDB(imSet);
save(sprintf(DATAopts.imdb, imSet), 'imdb');


%% GStructs

saveGStructs('train');

saveGStructs('val');

