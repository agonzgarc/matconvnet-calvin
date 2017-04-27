function downloadPASCALParts()
% downloadPASCALParts()
%
% Downloads and unpacks the PASCAL-Part dataset.
%

% Settings
zipNameData = 'trainval.tar.gz';
urlData = 'http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/trainval.tar.gz';
rootFolder = calvin_root();
datasetFolder = fullfile(rootFolder, 'data', 'Datasets', 'VOC2010');
downloadFolder = fullfile(rootFolder, 'data', 'Downloads');
zipFileData = fullfile(downloadFolder, zipNameData);
partsFolder = fullfile(datasetFolder, 'VOCdevkit','VOC2010');

% Download dataset
if ~exist(partsFolder, 'dir')
    % Create folder
    if ~exist(datasetFolder, 'dir')
        mkdir(datasetFolder);
    end
    if ~exist(downloadFolder, 'dir')
        mkdir(downloadFolder);
    end
    
    % Download tar file
    if ~exist(zipFileData, 'file')
        fprintf('Downloading PASCAL-Part dataset...\n');
        urlwrite(urlData, zipFileData);
    end

    % Untar it
    fprintf('Unpacking PASCAL-Part annotations...\n');
    untar(zipFileData, partsFolder);
   
end

% Add to path
addpath(devkitFolder);