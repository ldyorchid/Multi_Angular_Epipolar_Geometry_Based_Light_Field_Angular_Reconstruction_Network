classdef Permute5D_1 < dagnn.Layer
    %RESHAPE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        dims = 8
        sp2an = true
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            outputs = vl_nnpermute5D_1(inputs, obj.sp2an);
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs = vl_nnpermute5D_1(inputs, obj.sp2an, derOutputs);
            derParams = {} ;
        end
        
        function obj = Permute5D_1(varargin)
          obj.load(varargin);
        end
    end
    
end

