% Demo Joint object and part detection
%
% It prepares all the necessary structures and then trains and test several
% networks
%

% Train baseline?
baseline = true;

% Use offsetNet for the final model?
offsetNet = true;

% Add folders to path
setup();

% Download datasets
downloadVOC2010();

downloadPASCALParts();

% % Download base network
% downloadNetwork('modelName','imagenet-caffe-alex');
% 
% % Download Selective Search
% downloadSelectiveSearch();
% 
% % Create structures with part and object info
% setupParts();

times = zeros(3,2);

% Train and te2st baseline network
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





