% Demo Joint object and part detection
%
% It prepares all the necessary structures and then trains and test several
% networks
%

% Train baseline?
baseline = false;

% Use offsetNet for the final model?
offsetNet = false;

% If so, train coefficients?
trainCoeffs = false;

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
if baseline
    calvinNNPartDetection();
else
    dowloadModel('parts_baseline');
end

% Train and test our model, using as input baseline network
calvinNNPartDetectionObjAppCls();

% Train Offset Net
if offsetNet
    calvinNNOffsetNet(); 
end





