classdef ConvCombination < dagnn.Filter
    % Abel
    % Convolutional layer (expected to be used as fully connected) in which
    % the input spans the first two fields of the input struct
  properties
    size = [0 0 0 0]
    hasBias = true
    opts = {'cuDNN'}
    numInputs = 2;
  end

  methods
    function outputs = forward(obj, inputs, params)
      if ~obj.hasBias, params{2} = [] ; end
      
          % Take the object appearance of the boxes indicated by idxSupp
          objApp = inputs{3}(:,:,:,inputs{4});
          actualInput = cat(3, inputs{1}, inputs{2}, objApp);
     
          outputs{1} = vl_nnconv(...
           actualInput, params{1}, params{2}, ...
            'pad', obj.pad, ...
            'stride', obj.stride, ...
            obj.opts{:}) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      if ~obj.hasBias, params{2} = [] ; end

      objApp = inputs{3}(:,:,:,inputs{4});
      actualInput = cat(3, inputs{1}, inputs{2}, objApp);
      
      [tmpDerInputs, derParams{1}, derParams{2}] = vl_nnconv(...
       actualInput, params{1}, params{2}, derOutputs{1}, ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        obj.opts{:}) ;
       
        derInputs{1} = tmpDerInputs(:,:,1:size(inputs{1},3),:);

        derInputs{2} = tmpDerInputs(:,:,size(inputs{1},3)+1:size(inputs{1},3)+size(inputs{2},3),:);
        suppObjDerInputs = tmpDerInputs(:,:,size(inputs{1},3)+size(inputs{2},3)+1:end,:);
        idxSupp = squeeze(inputs{4});
        
        % Initialize with right dimension and type
        derInputs{3} = inputs{3}*0;

        % Accumulate gradients for supporting object proposals
        for ii = 1:max(idxSupp)
        derInputs{3}(:,:,:,ii) = derInputs{3}(:,:,:,ii) + ...
            sum(suppObjDerInputs(:,:,:,idxSupp == ii),4);
        end
        
        % We don't backprop on idxSupp
        derInputs{4} =[];
      
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.size(1:2) ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = obj.size(4) ;
    end

    function params = initParams(obj)
      sc = sqrt(2 / prod(obj.size(1:3))) ;
      params{1} = randn(obj.size,'single') * sc ;
      if obj.hasBias
        params{2} = zeros(obj.size(4),1,'single') * sc ;
      end
    end

    function set.size(obj, ksize)
      % make sure that ksize has 4 dimensions
      ksize = [ksize(:)' 1 1 1 1] ;
      obj.size = ksize(1:4) ;
    end

    function obj = ConvCombination(varargin)
      obj.load(varargin) ;
      % normalize field by implicitly calling setters defined in
      % dagnn.Filter and here
      obj.size = obj.size ;
      obj.stride = obj.stride ;
      obj.pad = obj.pad ;
      obj.numInputs = obj.numInputs;
    end
    
    function forwardAdvanced(obj, layer)
    %FORWARDADVANCED  Advanced driver for forward computation
    %  FORWARDADVANCED(OBJ, LAYER) is the advanced interface to compute
    %  the forward step of the layer.
    %
    %  The advanced interface can be changed in order to extend DagNN
    %  non-trivially, or to optimise certain blocks.

      in = layer.inputIndexes ;
      out = layer.outputIndexes ;
      par = layer.paramIndexes ;
      net = obj.net ;

      inputs = {net.vars(in).value} ;

      % give up if any of the inputs is empty (this allows to run
      % subnetworks by specifying only some of the variables as input --
      % however it is somewhat dangerous as inputs could be legitimaly
      % empty)
      if any(cellfun(@isempty, inputs)), return ; end

      % clear inputs if not needed anymore
      for v = in
        net.numPendingVarRefs(v) = net.numPendingVarRefs(v) - 1 ;
        if net.numPendingVarRefs(v) == 0
          if ~net.vars(v).precious & ~net.computingDerivative & net.conserveMemory
            net.vars(v).value = [] ;
          end
        end
      end

      %[net.vars(out).value] = deal([]) ;

      % call the simplified interface
      outputs = obj.forward(inputs, {net.params(par).value}) ;
      for oi = 1:numel(out)
        net.vars(out(oi)).value = outputs{oi};
      end
    end

    
    function backwardAdvanced(obj, layer)
    %BACKWARDADVANCED Advanced driver for backward computation
    %  BACKWARDADVANCED(OBJ, LAYER) is the advanced interface to compute
    %  the backward step of the layer.
    %
    %  The advanced interface can be changed in order to extend DagNN
    %  non-trivially, or to optimise certain blocks.
      in = layer.inputIndexes ;
      out = layer.outputIndexes ;
      par = layer.paramIndexes ;
      net = obj.net ;

      inputs = {net.vars(in).value} ;
      derOutputs = {net.vars(out).der} ;
      for i = 1:numel(derOutputs)
        if isempty(derOutputs{i}), return ; end
      end

      if net.conserveMemory
        % clear output variables (value and derivative)
        % unless precious
        for i = out
          if net.vars(i).precious, continue ; end
          net.vars(i).der = [] ;
          net.vars(i).value = [] ;
        end
      end

      % compute derivatives of inputs and paramerters
      [derInputs, derParams] = obj.backward ...
        (inputs, {net.params(par).value}, derOutputs) ;
    
      % accumuate derivatives
      for i = 1:numel(in)
        v = in(i) ;
        if net.numPendingVarRefs(v) == 0 || isempty(net.vars(v).der)
          net.vars(v).der = derInputs{i} ;
        elseif ~isempty(derInputs{i})
          net.vars(v).der = net.vars(v).der + derInputs{i} ;
        end
        net.numPendingVarRefs(v) = net.numPendingVarRefs(v) + 1 ;
      end

      for i = 1:numel(par)
        p = par(i) ;
        if (net.numPendingParamRefs(p) == 0 && ~net.accumulateParamDers) ...
              || isempty(net.params(p).der)
          net.params(p).der = derParams{i} ;
        else
          net.params(p).der = net.params(p).der + derParams{i} ;
        end
        net.numPendingParamRefs(p) = net.numPendingParamRefs(p) + 1 ;
      end
    end
  end
end
