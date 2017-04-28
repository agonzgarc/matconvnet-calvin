classdef SupportingSelector < dagnn.Layer
    % Abel
    % Selects supporting object proposals based on object scores and
    % insideness
    % inputs: object boxes, object scores for obj boxes, part boxes, and
    %         insideness
    %
    % outputs: object scores for supporting object proposals, for each of
    %          the part boxes
  
    properties
        threshInside = 0.9;
    end
    
    methods
    function outputs = forward(obj, inputs, params)
      
      % Get inputs
      assert(numel(inputs) == 4);
      scoresObj = squeeze(inputs{1})';
      boxesObj = squeeze(inputs{2})';
      boxesPrt = squeeze(inputs{3})';
      insideness = inputs{4};
      
      % Get for each prt box those obj boxes that contain them
      idxSupp = zeros(size(boxesPrt,1),1);
      for ii = 1:size(boxesPrt,1)
            objBoxesCont = insideness(ii,:) > obj.threshInside;
            
            if sum(objBoxesCont) == 0
                % No object box contains the part box enough --> take the 
                % one that contains it most
                [~,idxSupp(ii)] = max(insideness(ii,:));
            else
                idxObjBoxesCont = find(objBoxesCont);
                scoresObjCont = scoresObj(idxObjBoxesCont,:);
                
                % Ignore background score
                scoresObjCont(:,1) = 0;

                % Take max score over all classes
                scoresObjMax = max(scoresObjCont,[],2);

                % And then across all object boxes that contain it
                [~, idxMaxScore] = max(scoresObjMax);
                idxSupp(ii) = idxObjBoxesCont(idxMaxScore);
            end
      end

      outputs{1} = reshape(scoresObj(idxSupp,:)', [1 1 size(inputs{1},3) size(inputs{3},2)]);
      [uIdxSupp, ~, iC] = unique(idxSupp);
      
      % Also get boxes of supporting object proposal, simplified, and idx
      outputs{2} = boxesObj(uIdxSupp,:)';
      outputs{3} = iC;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        % There is no backpropagation in this layer
       derInputs{1} = [];
       derInputs{2} = [];
       derInputs{3} = [];
       derInputs{4} = [];

       derParams{1} = [];
    end

    function obj = SupportingSelector(varargin)
      obj.load(varargin) ;
    end

    
  end
end
