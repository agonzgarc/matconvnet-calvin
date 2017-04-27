% Demo Joint object and part detection
%
% It prepares all the necessary structures and then trains and test several
% networks
%

% Add folders to path
setup();

% Download datasets
downloadVOC2010();

downloadPASCALParts();

% Download base network
downloadNetwork('modelName','imagenet-caffe-alex');

% Download Selective Search
downloadSelectiveSearch();

% Create structures with part and object info
setupParts();

% Train and test baseline network

% Add 'if' here to decide whether train baseline net or download
calvinNNPartDetection();

% Train and test our model, using as input baseline network


