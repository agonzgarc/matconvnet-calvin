classdef ObjDetections < dagnn.Layer
    % 
    % Outputs set of object detections given all objects boxes and scores 
    % 
    % inputs: object boxes, object scores for obj boxes
    %
    % outputs: object detections (after min thresh and NMS) with scores
    %      
    % Created by Abel Gonzalez-Garcia, 2016
  
    properties
        minDetectionScore = 0.01;
        nmsTTest = 0.3;
    end
    
    methods
        function outputs = forward(obj, inputs, params)

          % Get inputs
          assert(numel(inputs) == 2);
          scoresObj = squeeze(inputs{1})';
          boxesObj = squeeze(inputs{2})';

          numObjClasses = size(scoresObj,2)-1;

          % Take at most 5 detections per object class
          objDets = zeros(5*numObjClasses,4,'like',boxesObj);
          objDetsScores = zeros(5*numObjClasses,1,'like',scoresObj);

          for idxClass = 1:numObjClasses
              scoresClass = scoresObj(:,idxClass+1);
              [currScoresT, sI] = sort(scoresClass, 'descend');
              currBoxes = boxesObj(sI,:);

              goodI = currScoresT > obj.minDetectionScore;
              currBoxes = currBoxes(goodI, :);
              currScoresT = currScoresT(goodI);
              
              [~, goodBoxesI] = BoxNMS(currBoxes, obj.nmsTTest);
              currBoxesT = currBoxes(goodBoxesI, :);
              currScoresT = currScoresT(goodBoxesI);

              numDets = size(currBoxesT,1);

              if numDets  == 0
                  % No detection over threshold, take highest scored box
                   [maxScore,idxMax] = max(scoresClass);
                   objDets((idxClass-1)*5+1:idxClass*5,:) = repmat(boxesObj(idxMax,:),5,1);
                   objDetsScores((idxClass-1)*5+1:idxClass*5) = [maxScore; ones(4,1)*-1];
              else
                   numDets = min(numDets,5);
                   objDets((idxClass-1)*5+1:(idxClass-1)*5+numDets,:) = currBoxesT(1:numDets,:);
                   objDetsScores((idxClass-1)*5+1:(idxClass-1)*5+numDets,:) = currScoresT(1:numDets,:);
                   % In case we had fewer than 5 detections --> replicate
                   % first but ignore it later
                   if numDets < 5
                       objDets((idxClass-1)*5+numDets+1:idxClass*5,:) = repmat(currBoxesT(1,:),5-numDets,1);
                       objDetsScores((idxClass-1)*5+numDets+1:idxClass*5) = -1*ones(5-numDets,1);
                   end
              end

          end

          % Prepare boxes dimensions to be input to roiPooling layer
          outputs{1} = objDets';
          outputs{2} = objDetsScores';
          
        end

        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            % There is no backpropagation in this layer
           derInputs{1} = [];
           derInputs{2} = [];

           derParams{1} = [];
        end

        function obj = ObjDetections(varargin)
          obj.load(varargin) ;
        end

  
  end
end
