classdef Concat_Reshape < dagnn.ElementWise
  properties
    dim = 4
  end

  properties (Transient)
    inputSizes = {}
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnconcat_reshape_3_7(inputs, obj.dim) ;
      %disp(size(outputs{1}));%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      obj.inputSizes = cellfun(@size, inputs, 'UniformOutput', false) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs = vl_nnconcat_reshape_3_7(inputs, obj.dim, derOutputs{1}, 'inputSizes', obj.inputSizes) ;
      %disp(size(derInputs{1}));%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
      %disp(size(derInputs{2}));%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
      derParams = {} ;
    end

    function reset(obj)
      obj.inputSizes = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      sz = inputSizes{1} ;
      for k = 2:numel(inputSizes)
        sz(obj.dim) = sz(obj.dim) + inputSizes{k}(obj.dim) ;
      end
      outputSizes{1} = sz ;
    end

    function rfs = getReceptiveFields(obj)
      numInputs = numel(obj.net.layers(obj.layerIndex).inputs) ;
      if obj.dim == 3 || obj.dim == 4
        rfs = getReceptiveFields@dagnn.ElementWise(obj) ;
        rfs = repmat(rfs, numInputs, 1) ;
      else
        for i = 1:numInputs
          rfs(i,1).size = [NaN NaN] ;
          rfs(i,1).stride = [NaN NaN] ;
          rfs(i,1).offset = [NaN NaN] ;
        end
      end
    end

    function load(obj, varargin)
      s = dagnn.Layer.argsToStruct(varargin{:}) ;
      % backward file compatibility
      if isfield(s, 'numInputs'), s = rmfield(s, 'numInputs') ; end
      load@dagnn.Layer(obj, s) ;
    end

    function obj = Concat_Reshape(varargin)
      obj.load(varargin{:}) ;
    end
  end
end
