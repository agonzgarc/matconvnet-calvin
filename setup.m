function setup()
% setup()
%
% Add Matconvnet, Matconvnet-FCN and Matconvnet-Calvin to Matlab path 
% and initialize global variables used by the demos.
%
% Copyright by Holger Caesar, 2016

% Define paths
root = fileparts(mfilename('fullpath'));
matconvnetFolder = fullfile(root, 'matconvnet', 'matlab');
matconvnetCalvinFolder = fullfile(root, 'matconvnet-calvin');

% Add matconvnet
addpath(matconvnetFolder);
vl_setupnn();

% Add matconvnet-calvin
addpath(genpath(matconvnetCalvinFolder));

% Define global variables
global glBaseFolder glDatasetFolder glFeaturesFolder;
glBaseFolder = fullfile(root, 'data');
glDatasetFolder = fullfile(glBaseFolder, 'Datasets');
glFeaturesFolder = fullfile(glBaseFolder, 'Features');