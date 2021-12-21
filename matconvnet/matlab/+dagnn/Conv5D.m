classdef Conv5D < dagnn.Filter3D
  properties
    size = [0 0 0 0 0]
    hasBias = true
    opts = {'nocuDNN'}

  end

  methods
    function outputs = forward(obj, inputs, params)
      if ~obj.hasBias, params{2} = [] ; end
      outputs{1} = mex_conv3d(...   %要求输入为4D数据，params{1}为5D的Filter，params{2}为biases个数。三个参数均要求是single类型数据
        inputs{1}, params{1}, params{2}, ...
        'pad', obj.pad, ...
        'stride', obj.stride) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      if ~obj.hasBias, params{2} = [] ; end
      [derInputs{1}, derParams{1}, derParams{2}] = mex_conv3d(...
        inputs{1}, params{1}, params{2}, derOutputs{1}, ...
        'pad', obj.pad, ...
        'stride', obj.stride) ;
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.size(1:4) ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(4) = obj.size(5) ;
    end

    function params = initParams(obj)
      % Xavier improved
      sc = sqrt(2 / prod(obj.size(1:4))) ;
      %sc = sqrt(2 / prod(obj.size([1 2 4]))) ;
      params{1} = randn(obj.size,'single') * sc ;
      if obj.hasBias
        params{2} = zeros(obj.size(5),1,'single') ;
      end
    end

    function set.size(obj, ksize)
      % make sure that ksize has 6 dimensions
      ksize = [ksize(:)' 1 1 1 1 1] ;
      obj.size = ksize(1:5) ;
    end

    function obj = Conv5D(varargin)
      obj.load(varargin) ;
      % normalize field by implicitly calling setters defined in
      % dagnn.Filter and here
      obj.size = obj.size ;
      obj.stride = obj.stride ;
      obj.pad = obj.pad ;
    end
  end
end
