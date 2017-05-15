classdef ImdbOffsetNet < ImdbMatbox
    properties(SetAccess = protected, GetAccess = public)
        negOverlapRange = [0 0.3];
        posOverlap = 0.7;
        boxesPerIm = 64;
        boxRegress = true;
        instanceWeighting = false;
        numClassesPrt = 1;
        numClassesObj = 1;
        numOuts = 1;
        idxPartGlobal2idxClass = [];
        % Object classes to be removed --> 4 for PASCAL-Part
        idxObjClassRem = [4 9 11 18];
    end
    methods
        function obj = ImdbOffsetNet(imageDir, imExt, matboxDir, filenames, datasetIdx, meanIm, numOuts, idxPartGlobal2idxClass, idxObjClassRem)
            obj@ImdbMatbox(imageDir, imExt, matboxDir, filenames, datasetIdx, meanIm);
            obj.minBoxSize = 10;
            gStruct = obj.LoadGStruct(1);
            obj.numClassesPrt = size(gStruct.overlapPrt, 2) + 1;
            obj.numClassesObj = size(gStruct.overlapObj, 2) + 1;
            obj.numOuts = numOuts;
            obj.idxPartGlobal2idxClass = idxPartGlobal2idxClass;
            obj.idxPartGlobal2idxClass(1) = [];
            obj.idxObjClassRem = idxObjClassRem;
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
            
            % Do not flip image for Offset Net
            
            if ismember(obj.datasetMode, {'train', 'val'})
               
                [objDets, dispWindowsTargets, presenceTargets] = obj.SamplePosAndNegFromGstruct(gStruct, obj.boxesPerIm);

                % Assign elements to cell array for use in training the network
                numElements = min(obj.boxesPerIm, size(objDets,1));
                
                numBatchFields = 2 * (2 + obj.boxRegress + obj.instanceWeighting);
                batchData = cell(numBatchFields, 1);
                idx = 1;
                batchData{idx} = 'input';       idx = idx + 1;
                batchData{idx} = image;         idx = idx + 1;
                batchData{idx} = 'objDets';       idx = idx + 1;
                batchData{idx} = objDets';        idx = idx + 1;
                for idxNumOut = 1:max(obj.numOuts)
                    batchData{idx} = ['dispWindows' num2str(idxNumOut) 'Targets'];       idx = idx + 1;
                    batchData{idx} = dispWindowsTargets(:,:,idxNumOut)';        idx = idx + 1;
                end
                batchData{idx} = 'presenceTargets';   idx = idx + 1;
                batchData{idx} = reshape(presenceTargets', 1,1,size(presenceTargets,2), size(presenceTargets,1));     idx = idx + 1;    
                batchData{idx} = 'oriImSize';   idx = idx + 1;
                batchData{idx} = oriImSize;    

            else
               % Test set. Get all boxes
                [boxesPrt, keysPrt] = obj.SampleAllBoxesFromGstructPrt(gStruct);
                [boxesObj, keysObj] = obj.SampleAllBoxesFromGstructObj(gStruct);

                numElements = max(size(boxesPrt,1),size(boxesObj,1));
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
            % image = LoadImage(obj, batchIdx)
            % Loads an image from disk, resizes it, and subtracts the mean image
            imageT = single(imread([obj.imageDir obj.data.(obj.datasetMode){batchIdx} obj.imExt]));
            oriImSize = double(size(imageT));
            if numel(obj.meanIm) == 3
                for colourI = 1:3
                    imageT(:,:,colourI) = imageT(:,:,colourI) - obj.meanIm(colourI);
                end
            else
                imageT = imageT - imresize(obj.meanIm, [oriImSize(1) oriImSize(2)]); % Subtract mean im
            end
            
            resizeFactor = 1000 / max(oriImSize(1:2));
%             resizeFactorMin = 600 / min(oriImSize(1:2));
%             resizeFactor = min(resizeFactorMin, resizeFactorMax);
            if gpuMode
                image = gpuArray(imageT);
                image = imresize(image, resizeFactor);
            else
                image = imresize(imageT, resizeFactor, 'bilinear', 'antialiasing', false);
            end
            
%             % Subtract mean image
%             meanIm = imresize(obj.meanIm, [size(image,1) size(image,2)], 'bilinear', 'antialiasing', false);
%             if gpuMode
%                 meanIm = gpuArray(meanIm);
%             end
%             image = image - meanIm;
        end
        
        
        function [objDets, dispWindowsTargets, presenceTargets] = SamplePosAndNegFromGstruct(obj, gStruct, numSamples)
             
            allBoxes = gStruct.boxes(gStruct.boxesObj,:);
            overlap = gStruct.overlapObj(gStruct.boxesObj,:);
            class = gStruct.classObj(gStruct.boxesObj,:);

            
            % Get positive, negative, and true GT keys
            [maxOverlap, idxObjClassOv] = max(overlap, [], 2);

            posKeys = find(maxOverlap >=  obj.posOverlap & class == 0);
            
            % Remove those classes not             
            idxGT = class>0 & ~ismember(class,obj.idxObjClassRem);
              
            % Get correct number of positives
            numExtraPos = numSamples - sum(idxGT);
            numExtraPos = min(numExtraPos, length(posKeys));
            if numExtraPos > 0
                posKeys = posKeys(randperm(length(posKeys), numExtraPos));
            else
               posKeys = [];
            end
            
            objDets = [allBoxes(idxGT,:); allBoxes(posKeys,:)];
            objClassPos = [class(idxGT); idxObjClassOv(posKeys)];

            mapNumOuts = [1; cumsum(obj.numOuts)+1];
            mapNumOuts(end) = [];
            totalPresenceTargets = sum(obj.numOuts);
            
            presenceTargets = -1*ones(size(objDets,1), totalPresenceTargets, 'like', objDets);
            
            % Create NaN array: nans represent numbers which will not be active
            % in regression
            dispWindowsTargets = nan([size(objDets,1) 4 * (obj.numClassesPrt-1) max(obj.numOuts)], 'like', objDets);

            keys = [find(idxGT); posKeys];

            % Get GT parts boxes here
            prtBoxes = gStruct.boxes(gStruct.boxesPrt,:);
            prtClass = gStruct.classPrt(gStruct.boxesPrt,:);
            % Take subset of insideness to preserve indexing
            % Rows are prt boxes, columns are object boxes
            prtInsideness = gStruct.insideness(gStruct.boxesPrt,gStruct.boxesObj);
            gtKeysPrt = find(prtClass > 0);

            gtBoxes = prtBoxes(gtKeysPrt,:);
            gtClasses = prtClass(gtKeysPrt);
            gtObjClasses = obj.idxPartGlobal2idxClass(gtClasses);
            
            
            for kk = 1:size(keys,1)
                 
                % We should only go through parts that belong to the object
                for idxPart = 1:obj.numClassesPrt-1

                    isPartInBox = gtClasses == idxPart & prtInsideness(gtKeysPrt,  keys(kk)) > 0.9 ...
                        & gtObjClasses == objClassPos(kk);

                    numPartsInBox = sum(isPartInBox);
                    
                    if  numPartsInBox > 0
                        targetRangeBegin = 4 * (idxPart-1)+1;
                        targetRange = targetRangeBegin:(targetRangeBegin+3);

                        idxBoxPart = find(isPartInBox); 

                        if obj.numOuts(idxPart) == 1

                            idxBoxPart = idxBoxPart(randi(numPartsInBox));
                            partBox = gtBoxes(idxBoxPart,:);

                            dispWindowsTargets(kk,targetRange,1) = BoxRegressionTargetGirshick(partBox, objDets(kk,:));
                            presenceTargets(kk, mapNumOuts(idxPart)) = 1;

                        else
                            a = objDets(kk,1);
                            b = objDets(kk,3);

                            limRegions = a:floor((b-a)/obj.numOuts(idxPart)):b;

                            % Force it to finish at the very end
                            limRegions(end) = b;

                            partBoxesCX = round(gtBoxes(isPartInBox,1)+gtBoxes(isPartInBox,3))/2;

                            % Assign boxes accordingly
                            for idxNumOut = 1:obj.numOuts(idxPart)
                               % Get boxes of corresponding region
                               idxBoxesInRegion = limRegions(idxNumOut) <= partBoxesCX & partBoxesCX < limRegions(idxNumOut + 1);
                               partBoxesRegion = gtBoxes(idxBoxPart(idxBoxesInRegion),:);
                               
                               if size(partBoxesRegion,1)>0
                                    dispWindowsTargets(kk,targetRange,idxNumOut) = BoxRegressionTargetGirshick(partBoxesRegion(randi(size(partBoxesRegion,1)),:), objDets(kk,:));
                                    presenceTargets(kk, mapNumOuts(idxPart) + idxNumOut-1) = 1;
                               end

                            end
                        end

                    end


                end
               
             end
             
            % Some of the object "detections" don't have any part inside
            % --> remove them, they affect negatively to presence
            idxEmpty = sum(presenceTargets==1,2) == 0;
            objDets = objDets(~idxEmpty,:);
            dispWindowsTargets = dispWindowsTargets(~idxEmpty,:,:);
            presenceTargets = presenceTargets(~idxEmpty,:);
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
