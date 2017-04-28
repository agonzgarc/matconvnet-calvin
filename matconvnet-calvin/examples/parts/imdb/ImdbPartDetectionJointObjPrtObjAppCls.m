classdef ImdbPartDetectionJointObjPrtObjAppCls < ImdbMatbox
    properties(SetAccess = protected, GetAccess = public)
        negOverlapRange = [0.1 0.4];
        posOverlap = 0.4;
        boxesPerIm = 128;
        boxRegress = true;
        instanceWeighting = false;
        numClassesPrt = 1;
        numClassesObj = 1;
    end
    methods
        function obj = ImdbPartDetectionJointObjPrtObjAppCls(imageDir, imExt, matboxDir, filenames, datasetIdx, meanIm)
            obj@ImdbMatbox(imageDir, imExt, matboxDir, filenames, datasetIdx, meanIm);
            obj.minBoxSize = 10;
            gStruct = obj.LoadGStruct(1);
            obj.numClassesPrt = size(gStruct.overlapPrt, 2) + 1;
            obj.numClassesObj = size(gStruct.overlapObj, 2) + 1;
        end
        
        function [batchData, numElements] = getBatch(obj, batchInds, net, ~)
            if length(batchInds) > 1
                error('Only supports batches of 1');
            end
            
            if nargin == 2
                gpuMode = false;
            else
                gpuMode = strcmp(net.device, 'gpu');
            end
            
            % Load image. Make correct size. Subtract average im.
            [image, oriImSize] = obj.LoadImage(batchInds, gpuMode);

            % Sample boxes
            gStruct = obj.LoadGStruct(batchInds);
            
            % Flip the image and boxes at training time
            % Note: flipLR alternates between true and false in ImdbMatbox.initEpoch()
            if obj.flipLR && strcmp(obj.datasetMode, 'train')
                currImT = fliplr(image);
                currBoxesT = gStruct.boxes;
                currBoxesT(:,3) = oriImSize(2) - gStruct.boxes(:,1) + 1;
                currBoxesT(:,1) = oriImSize(2) - gStruct.boxes(:,3) + 1;
                gStruct.boxes = currBoxesT;
                image = currImT;
            end
            
            if ismember(obj.datasetMode, {'train', 'val'})
                [boxesPrt, labelsPrt, overlapScoresPrt, keysPrt, regressionFactorsPrt] = obj.SamplePosAndNegFromGstructPrt(gStruct, obj.boxesPerIm);
                [boxesObj, labelsObj, overlapScoresObj, keysObj, regressionFactorsObj] = obj.SamplePosAndNegFromGstructObj(gStruct, obj.boxesPerIm);

                % Assign elements to cell array for use in training the network
                numElements = min(obj.boxesPerIm, size(boxesPrt,1));
                
                % Input image and size are shared for Obj and Prt
                numBatchFields =  4 + 4 * (2 + obj.boxRegress + obj.instanceWeighting);
                batchData = cell(numBatchFields, 1);
                idx = 1;
                batchData{idx} = 'input';       idx = idx + 1;
                batchData{idx} = image;         idx = idx + 1;
                batchData{idx} = 'labelPrt';       idx = idx + 1;
                batchData{idx} = labelsPrt';       idx = idx + 1;
                batchData{idx} = 'boxesPrt';       idx = idx + 1;
                batchData{idx} = boxesPrt';        idx = idx + 1;
                batchData{idx} = 'labelObj';       idx = idx + 1;
                batchData{idx} = labelsObj';       idx = idx + 1;
                batchData{idx} = 'boxesObj';       idx = idx + 1;
                batchData{idx} = boxesObj';        idx = idx + 1;
                batchData{idx} = 'oriImSize';   idx = idx + 1;
                batchData{idx} = oriImSize;     idx = idx + 1;
                batchData{idx} = 'insideness';   idx = idx + 1;
                batchData{idx} = gStruct.insideness(keysPrt,keysObj);     idx = idx + 1;
                
                if obj.boxRegress
                    batchData{idx} = 'regressionTargetsPrt';   idx = idx + 1;
                    batchData{idx} = regressionFactorsPrt';    idx = idx + 1;                    
                    batchData{idx} = 'regressionTargetsObj';   idx = idx + 1;
                    batchData{idx} = regressionFactorsObj';    idx = idx + 1;       
                end
                if obj.instanceWeighting
                    instanceWeightsPrt = overlapScoresPrt;
                    instanceWeightsPrt(labelsPrt == 1) = 1;
                    instanceWeightsPrt = reshape(instanceWeightsPrt, [1 1 1 length(instanceWeightsPrt)]); % VL-Feat way 
                    batchData{idx} = 'instanceWeightsPrt';     idx = idx + 1;
                    batchData{idx} = instanceWeightsPrt;       %idx = idx + 1;
                    instanceWeightsObj = overlapScoresObj;
                    instanceWeightsObj(labelsObj == 1) = 1;
                    instanceWeightsObj = reshape(instanceWeightsObj, [1 1 1 length(instanceWeightsObj)]); % VL-Feat way
                    batchData{idx} = 'instanceWeightsObj';     idx = idx + 1;
                    batchData{idx} = instanceWeightsObj;       %idx = idx + 1;
                end
                
            else
                % Test set. Get all boxes
                [boxesPrt, keysPrt] = obj.SampleAllBoxesFromGstructPrt(gStruct);
                [boxesObj, keysObj] = obj.SampleAllBoxesFromGstructObj(gStruct);

                numElements = size(gStruct.boxes,1);
                batchData{10} = gStruct.insideness(keysPrt,keysObj);
                batchData{9} = 'insideness';
                batchData{8} = oriImSize;
                batchData{7} = 'oriImSize';
                batchData{6} = boxesObj';
                batchData{5} = 'boxesObj';
                batchData{4} = boxesPrt';
                batchData{3} = 'boxesPrt';
                batchData{2} = image;
                batchData{1} = 'input';
            end
            
        end
        
        function [image, oriImSize] = LoadImage(obj, batchIdx, gpuMode)
            % Loads an image from disk, resizes it, and subtracts the mean image
            imageT = single(imread([obj.imageDir obj.data.(obj.datasetMode){batchIdx} obj.imExt]));
            oriImSize = double(size(imageT));
            
             % Black and white image
            if numel(oriImSize) == 2
                imageT = cat(3, imageT,imageT,imageT);
                oriImSize = double(size(imageT));
            end
            
            if numel(obj.meanIm) == 3
                for colourI = 1:3
                    imageT(:,:,colourI) = imageT(:,:,colourI) - obj.meanIm(colourI);
                end
            else
                imageT = imageT - imresize(obj.meanIm, [oriImSize(1) oriImSize(2)]); % Subtract mean im
            end
            
            resizeFactor = 1000 / max(oriImSize(1:2));
            
            if gpuMode
                image = gpuArray(imageT);
                image = imresize(image, resizeFactor);
            else
                image = imresize(imageT, resizeFactor, 'bilinear', 'antialiasing', false);
            end
            
        end
        
        
        function [boxes, labels, overlapScores, genKeys, regressionTargets] = SamplePosAndNegFromGstructPrt(obj, gStruct, numSamples)
            
            % Consider only boxes for parts          
            allBoxes = gStruct.boxes(gStruct.boxesPrt,:);
            overlap = gStruct.overlapPrt(gStruct.boxesPrt,:);
            class = gStruct.classPrt(gStruct.boxesPrt,:);
            numClasses = obj.numClassesPrt;

            % Get positive, negative, and true GT keys
            [maxOverlap, classOverlap] = max(overlap, [], 2);

            posKeys = find(maxOverlap >= obj.posOverlap & class == 0);
            negKeys = find(maxOverlap < obj.negOverlapRange(2) & maxOverlap >= obj.negOverlapRange(1) & class == 0);
            gtKeys = find(class > 0);
            
            % If there are more gtKeys than the fraction of positives, just
            % take a subset
            if length(gtKeys) > numSamples * obj.posFraction
                gtKeys = gtKeys(randperm(length(gtKeys), numSamples * obj.posFraction));
            end

            % Get correct number of positive and negative samples
            numExtraPos = numSamples * obj.posFraction - length(gtKeys);
            numExtraPos = min(numExtraPos, length(posKeys));
            if numExtraPos > 0
                posKeys = posKeys(randperm(length(posKeys), numExtraPos));
            else
               numExtraPos = 0;
               posKeys = [];
            end
            numNeg = numSamples - numExtraPos - length(gtKeys);
            numNeg = min(numNeg, length(negKeys));
            negKeys = negKeys(randperm(length(negKeys), numNeg));

            % Concatenate for final keys and labs
            keys = cat(1, gtKeys, posKeys, negKeys);
            labels = cat(1, class(gtKeys), classOverlap(posKeys), zeros(numNeg, 1));
            labels = single(labels + 1); % Add 1 for background class
            boxes = allBoxes(keys,:);
            
            overlapScores = cat(1, ones(length(gtKeys),1), maxOverlap(posKeys), maxOverlap(negKeys));
            
            % Get overall keys for insideness info
            genKeys = find(gStruct.boxesPrt);
            genKeys = genKeys(keys);
            
            % Calculate regression targets.
            % Jasper: I simplify Girshick by implementing regression through four
            % scalars which scale the box with respect to its center.
            if nargout == 5
                % Create NaN array: nans represent numbers which will not be active
                % in regression
                regressionTargets = nan([size(boxes,1) 4 * numClasses], 'like', boxes);
                
                % Get scaling factors for all positive boxes
                gtBoxes = allBoxes(gtKeys,:);
                for bI = 1:length(gtKeys)+length(posKeys)
                    % Get current box and corresponding GT box
                    currPosBox = boxes(bI,:);
                    [~, gtI] = BoxBestOverlapFastRcnn(gtBoxes, currPosBox);
                    currGtBox = gtBoxes(gtI,:);
                    
                    % Get range of regression target based on the label of the gt box
                    targetRangeBegin = 4 * (labels(bI)-1)+1;
                    targetRange = targetRangeBegin:(targetRangeBegin+3);
                    
                    % Set regression targets
                    regressionTargets(bI,targetRange) = BoxRegressionTargetGirshick(currGtBox, currPosBox);
                    
                end
            end 
        end
        
        
        function [boxes, labels, overlapScores, genKeys, regressionTargets] = SamplePosAndNegFromGstructObj(obj, gStruct, numSamples)
            
            % Consider only boxes for parts          
            allBoxes = gStruct.boxes(gStruct.boxesObj,:);
            overlap = gStruct.overlapObj(gStruct.boxesObj,:);
            class = gStruct.classObj(gStruct.boxesObj,:);
            numClasses = obj.numClassesObj;

            % Get positive, negative, and true GT keys
            [maxOverlap, classOverlap] = max(overlap, [], 2);

            posKeys = find(maxOverlap >= obj.posOverlap & class == 0);
            negKeys = find(maxOverlap < obj.negOverlapRange(2) & maxOverlap >= obj.negOverlapRange(1) & class == 0);
            gtKeys = find(class > 0);
            
            % If there are more gtKeys than the fraction of positives, just
            % take a subset
            if length(gtKeys) > numSamples * obj.posFraction
                gtKeys = gtKeys(randperm(length(gtKeys), numSamples * obj.posFraction));
            end

            % Get correct number of positive and negative samples
            numExtraPos = numSamples * obj.posFraction - length(gtKeys);
            numExtraPos = min(numExtraPos, length(posKeys));
            if numExtraPos > 0
                posKeys = posKeys(randperm(length(posKeys), numExtraPos));
            else
               numExtraPos = 0;
               posKeys = [];
            end
            numNeg = numSamples - numExtraPos - length(gtKeys);
            numNeg = min(numNeg, length(negKeys));
            negKeys = negKeys(randperm(length(negKeys), numNeg));

            % Concatenate for final keys and labs
            keys = cat(1, gtKeys, posKeys, negKeys);
            labels = cat(1, class(gtKeys), classOverlap(posKeys), zeros(numNeg, 1));
            labels = single(labels + 1); % Add 1 for background class
            boxes = allBoxes(keys,:);
            
            overlapScores = cat(1, ones(length(gtKeys),1), maxOverlap(posKeys), maxOverlap(negKeys));

            % Get overall keys for insideness info
            genKeys = find(gStruct.boxesObj);
            genKeys = genKeys(keys);
            
            % Calculate regression targets.
            % Jasper: I simplify Girshick by implementing regression through four
            % scalars which scale the box with respect to its center.
            if nargout == 5
                % Create NaN array: nans represent numbers which will not be active
                % in regression
                regressionTargets = nan([size(boxes,1) 4 * numClasses], 'like', boxes);
                
                % Get scaling factors for all positive boxes
                gtBoxes = allBoxes(gtKeys,:);
                for bI = 1:length(gtKeys)+length(posKeys)
                    % Get current box and corresponding GT box
                    currPosBox = boxes(bI,:);
                    [~, gtI] = BoxBestOverlapFastRcnn(gtBoxes, currPosBox);
                    currGtBox = gtBoxes(gtI,:);
                    
                    % Get range of regression target based on the label of the gt box
                    targetRangeBegin = 4 * (labels(bI)-1)+1;
                    targetRange = targetRangeBegin:(targetRangeBegin+3);
                    
                    % Set regression targets
                    regressionTargets(bI,targetRange) = BoxRegressionTargetGirshick(currGtBox, currPosBox);
                    
                end
            end 
        end
        
        
        function [boxes, keys] = SampleAllBoxesFromGstructPrt(~,gStruct)
            boxes = gStruct.boxes(gStruct.boxesPrt,:);
            keys = find(gStruct.boxesPrt);
        end

        function [boxes, keys] = SampleAllBoxesFromGstructObj(~,gStruct)
            boxes = gStruct.boxes(gStruct.boxesObj,:);
            keys = find(gStruct.boxesObj);
        end
        
        % Load gStruct
        function gStruct = LoadGStruct(obj,imI)
            gStruct = load([obj.matBoxDir obj.data.(obj.datasetMode){imI} '.mat']);
            
            % Make sure that no GT boxes/labels/etc are given when using test phase
            if strcmp(obj.datasetMode, 'test')
                goodIds = ~(gStruct.gtPrt | gStruct.gtObj);
                gStruct.gtPrt = gStruct.gtPrt(goodIds,:);
                gStruct.gtObj = gStruct.gtObj(goodIds,:);

                gStruct.overlapPrt = gStruct.overlapPrt(goodIds,:);
                gStruct.overlapObj = gStruct.overlapObj(goodIds,:);

                gStruct.boxes = gStruct.boxes(goodIds,:);
                gStruct.boxesPrt = gStruct.boxesPrt(goodIds,:);
                gStruct.boxesObj = gStruct.boxesObj(goodIds,:);

                gStruct.classPrt = gStruct.classPrt(goodIds,:);
                gStruct.classObj = gStruct.classObj(goodIds,:);

                gStruct.insideness = gStruct.insideness(goodIds,goodIds);
            end
            
            % Remove small boxes
            [nR, nC] = BoxSize(gStruct.boxes);
            badI = ((nR < obj.minBoxSize) | (nC < obj.minBoxSize)) & ~(gStruct.gtPrt | gStruct.gtObj) ;
            gStruct.gtPrt = gStruct.gtPrt(~badI,:);
            gStruct.gtObj = gStruct.gtObj(~badI,:);

            gStruct.overlapPrt = gStruct.overlapPrt(~badI,:);
            gStruct.overlapObj = gStruct.overlapObj(~badI,:);

            gStruct.boxes = gStruct.boxes(~badI,:);
            gStruct.boxesPrt = gStruct.boxesPrt(~badI,:);
            gStruct.boxesObj = gStruct.boxesObj(~badI,:);

            gStruct.classPrt = gStruct.classPrt(~badI,:);
            gStruct.classObj = gStruct.classObj(~badI,:);

            gStruct.insideness = gStruct.insideness(~badI,~badI);
            
            % Copy one of the overlap vars into 'overlap' for ImdbMatbox
            % initialization of number of classes
            gStruct.overlap = gStruct.overlapObj;
        end
        
        function SetBoxRegress(obj, doRegress)
            obj.boxRegress = doRegress;
        end
        
        function SetInstanceWeighting(obj, doInstanceWeighting)
            obj.instanceWeighting = doInstanceWeighting;
        end
        
        function SetPosOverlap(obj, posOverlap)
            obj.posOverlap = posOverlap;
        end
        
        function SetNegOverlapRange(obj, negOverlapRange)
            obj.negOverlapRange = negOverlapRange;
        end
        
    end % End methods
end % End classdef
