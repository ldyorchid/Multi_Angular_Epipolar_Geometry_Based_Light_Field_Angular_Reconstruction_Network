classdef Filter3D < dagnn.Layer
  properties
    pad = [0 0 0 0 0 0] %��ά
    stride = [1 1 1] %��ά

  end
  methods
    function set.pad(obj, pad)
      if numel(pad) == 1
        obj.pad = [pad pad pad pad pad pad] ; %��ά
      elseif numel(pad) == 2
        obj.pad = pad([1 1 1 2 2 2]) ; %��ά
      else
        obj.pad = pad ;
      end
    end

    function set.stride(obj, stride)
      if numel(stride) == 1
        obj.stride = [stride stride stride] ;   %��ά
      else
        obj.stride = stride ;
      end
    end
    

    function kernelSize = getKernelSize(obj)
      kernelSize = [1 1 1] ;   %��ά�����
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      ke = obj.getKernelSize() ;
      outputSizes{1} = [...
        fix((inputSizes{1}(1) + obj.pad(1) + obj.pad(2) - ke(1)) / obj.stride(1)) + 1, ...
        fix((inputSizes{1}(2) + obj.pad(3) + obj.pad(4) - ke(2)) / obj.stride(2)) + 1, ...
        fix((inputSizes{1}(3) + obj.pad(5) + obj.pad(6) - ke(3)) / obj.stride(3)) + 1, ...
        1, ...
        inputSizes{1}(5)] ;   %��ά����ˣ�����5��ά��
    end    

    function rfs = getReceptiveFields(obj)
      ke = obj.getKernelSize() ;
      y1 = 1 - obj.pad(1) ;
      y2 = 1 - obj.pad(1) + ke(1) - 1 ;
      x1 = 1 - obj.pad(3) ;
      x2 = 1 - obj.pad(3) + ke(2) - 1 ;
      z1 = 1 - obj.pad(5) ;
      z2 = 1 - obj.pad(5) + ke(3) - 1 ;
      h = y2 - y1 + 1 ;
      w = x2 - x1 + 1 ;
      d = z2 - z1 + 1 ;
         
      rfs.size = [h, w, d] ;
      rfs.stride = obj.stride ;
      rfs.strideAngular = obj.strideAngular ;
      rfs.offset = [y1+y2, x1+x2, z1+z2]/2 ;
    end
  end
end
