function inputs = getBatch(opts, imdb, batch, mode)
% -------------------------------------------------------------------------
%   Description:
%       get one batch for training LapSRN
%       Modified from the code produced by the authors in the citation below
%
%   Input:
%       - opts  : options generated from init_opts()
%       - imdb  : imdb file generated from make_imdb()
%       - batch : array of ID to fetch
%       - mode  : 'train' or 'val'
%
%   Output:
%       - inputs: input for dagnn (include LR and HR images)
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

    %% get images
    image_batch = imdb.images.img(batch);  %376*541*64
    
    %% crop
    HR = zeros(opts.patch_size, opts.patch_size, opts.conv_a, length(batch), 'single');
    
    for i = 1:length(batch)
        
        img = image_batch{i};
              
        H = size(img, 1);
        W = size(img, 2);
        
        % random crop
        r1 = floor(opts.patch_size / 2);
        r2 = opts.patch_size - r1 - 1;
        
        mask = zeros(H, W);
        mask(1 + r1 : end - r2, 1 + r1 : end - r2) = 1;
        
        [X, Y] = meshgrid(1:W, 1:H);
        X = X(mask == 1);
        Y = Y(mask == 1);
        
        select = randperm(length(X), 1);
        X = X(select);
        Y = Y(select);

        HR(:, :, :, i) = img(Y - r1 : Y + r2, X - r1 : X + r2, :);
    end
 
    if ((randi(2) - 1)>0)
        HRIn = HR(:,:,[2:3,5:6,8:21,23:24,26:27,29:42,44:45,47:48],:);      
        HRIn = reshape(HRIn, [64,64,1,40]);
 
        LR_In = HR(:,:,[1,4,7,22,25,28,43,46,49],:);
        LR_In = reshape(LR_In, [64,64,9,1]);
        
        LR_Angle_0   = HR(:,:,[22,25,28],:);
        LR_Angle_45  = HR(:,:,[1,25,49],:);
        LR_Angle_90  = HR(:,:,[4,25,46],:);
        LR_Angle_135 = HR(:,:,[7,25,43],:);
        
        LR_Angle_0   = reshape(LR_Angle_0,  [64,64,3,1]);
        LR_Angle_45  = reshape(LR_Angle_45, [64,64,3,1]);
        LR_Angle_90  = reshape(LR_Angle_90, [64,64,3,1]);
        LR_Angle_135 = reshape(LR_Angle_135,[64,64,3,1]);
 
        

    else

        HRIn = HR(:,:,[2:3,5:6,8:21,23:24,26:27,29:42,44:45,47:48],:);
        HRIn = reshape(HRIn, [64,64,1,40]);
                
        LR_In = HR(:,:,[1,4,7,22,25,28,43,46,49],:);
        LR_In = reshape(LR_In, [64,64,9,1]);  
        
        LR_Angle_0   = HR(:,:,[22,25,28],:);
        LR_Angle_45  = HR(:,:,[1,25,49],:);
        LR_Angle_90  = HR(:,:,[4,25,46],:);
        LR_Angle_135 = HR(:,:,[7,25,43],:);
        
        LR_Angle_0   = reshape(LR_Angle_0,  [64,64,3,1]);
        LR_Angle_45  = reshape(LR_Angle_45, [64,64,3,1]);
        LR_Angle_90  = reshape(LR_Angle_90, [64,64,3,1]);
        LR_Angle_135 = reshape(LR_Angle_135,[64,64,3,1]);
        

    end

    %% make dagnn input
    inputs = {};
    inputs{end+1} = 'HR';
    inputs{end+1} = HRIn;
        
    inputs{end+1} = 'LR';
    inputs{end+1} = LR_In;
    
    inputs{end+1} = 'LR_Angle_0';
    inputs{end+1} =  LR_Angle_0;
    
    inputs{end+1} = 'LR_Angle_45';
    inputs{end+1} =  LR_Angle_45;
    
    inputs{end+1} = 'LR_Angle_90';
    inputs{end+1} =  LR_Angle_90;
    
    inputs{end+1} = 'LR_Angle_135';
    inputs{end+1} =  LR_Angle_135;
    

     
    % convert to GPU array
    if( opts.gpu > 0 )
        for i = 2:2:length(inputs)
            inputs{i} = gpuArray(inputs{i});
        end
    end
end
