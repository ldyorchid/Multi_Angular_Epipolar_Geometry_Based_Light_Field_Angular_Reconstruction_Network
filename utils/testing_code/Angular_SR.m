function img_HR = Angular_SR(img_LR, net, opts)
% -------------------------------------------------------------------------
%   Description:
%       function to apply the Angular SR model
%       Modified from the code produced by the authors in the citation below
%
%   Input:
%       - img_LR: low-resolution image
%       - net   : LapSRN model
%       - opts  : options generated from init_opts()
%
%   Output:
%       - img_HR: high-resolution image
%
%   Citation: 
%       Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution
%       Wei-Sheng Lai, Jia-Bin Huang, Narendra Ahuja, and Ming-Hsuan Yang
%       IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017
%
%   Contact:
%       Wei-Sheng Lai
%       wlai24@ucmerced.edu
%       University of California, Merced
% -------------------------------------------------------------------------

    %% setup
    net.mode = 'test' ;
    output_var = 'output';
    output_index = net.getVarIndex(output_var);
    
    % For pretrained models
    if isnan(output_index)
	    output_var = { 'level1_output' };
	    output_index = net.getLayerIndex(output_var) + 1;
    end
    
    net.vars(output_index).precious = 1;
    
    LR_In = img_LR(:,:,:,[1,4,7,22,25,28,43,46,49]);
    LR_In = reshape(LR_In, [size(LR_In, 1), size(LR_In, 2),9,1]);   
    LR_In = single(LR_In);
    
    LR_Angle_0   = img_LR(:,:,:,[22,25,28]);
    LR_Angle_45  = img_LR(:,:,:,[1,25,49]);
    LR_Angle_90  = img_LR(:,:,:,[4,25,46]);
    LR_Angle_135 = img_LR(:,:,:,[7,25,43]);
        
    LR_Angle_0   = reshape(LR_Angle_0,  [size(LR_Angle_0, 1), size(LR_Angle_0, 2),3,1]);
    LR_Angle_45  = reshape(LR_Angle_45, [size(LR_Angle_45, 1), size(LR_Angle_45, 2),3,1]);
    LR_Angle_90  = reshape(LR_Angle_90, [size(LR_Angle_90, 1), size(LR_Angle_90, 2),3,1]);
    LR_Angle_135 = reshape(LR_Angle_135,[size(LR_Angle_135, 1), size(LR_Angle_135, 2),3,1]);
    
    LR_Angle_0   = single(LR_Angle_0);
    LR_Angle_45  = single(LR_Angle_45);
    LR_Angle_90  = single(LR_Angle_90);
    LR_Angle_135 = single(LR_Angle_135);
    

    if( opts.gpu )
        LR_In        = gpuArray(LR_In);
        LR_Angle_0   = gpuArray(LR_Angle_0);
        LR_Angle_45  = gpuArray(LR_Angle_45);
        LR_Angle_90  = gpuArray(LR_Angle_90);
        LR_Angle_135 = gpuArray(LR_Angle_135);

    else
        LR_In        = LR_In;
        LR_Angle_0   = LR_Angle_0;
        LR_Angle_45  = LR_Angle_45;
        LR_Angle_90  = LR_Angle_90;
        LR_Angle_135 = LR_Angle_135;

    end
    
    % forward
    inputs = {'LR', LR_In, 'LR_Angle_0',LR_Angle_0,'LR_Angle_45',LR_Angle_45,'LR_Angle_90',LR_Angle_90,'LR_Angle_135',LR_Angle_135 };
   
    net.eval(inputs);
    if( opts.gpu )
        y = gather(net.vars(output_index).value);
    else
        y = net.vars(output_index).value;
    end    
    
    img_HR = double(y);
    crop = 8;
    img_HR = img_HR(1+crop:end-crop,1+crop:end-crop,:);
    
end
