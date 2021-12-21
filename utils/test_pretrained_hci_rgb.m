function test_pretrained_hci_rgb(depth, gpu, len, saveImg)
  
% -------------------------------------------------------------------------
%   Description:
%       Script to test pretrained Angular SR models
%
%   Input:
%       - depth : numbers of conv layers in each pyramid level
%       - gpu   : GPU ID, 0 for CPU mode
%       - len   : controls the size of the sub-lightfield, value depends on GPU memory
%       - imdb  : the loaded dataset
%
% -------------------------------------------------------------------------

    %% setup paths
    addpath(genpath('utils/IO_code'));
    addpath(genpath('utils/testing_code'));
    addpath(genpath('utils/training_code'));
    addpath(fullfile(pwd, 'matconvnet/matlab'));
    vl_setupnn;

    %% generate opts
    opts = init_opts('', depth, gpu);
    crop = 8; % the overlap between image patches
    
    %% Load model
    model_filename = ['pretrained_models/Layer' num2str(depth) '.mat'];
    fprintf('Load %s\n', model_filename);
    net = load(model_filename);
    net = dagnn.DagNN.loadobj(net.net);
    if (opts.gpu)
        gpuDevice(opts.gpu);
        net.move('gpu');
    end

    if(saveImg && ~exist(['Save_Img/Layer' num2str(depth)]))
        mkdir(strcat(['Save_Img/Layer' num2str(depth)]))
    end

    
     %% Load the HCI data
     sub_HCI_dir  = dir( opts.test_HCI_dir );
     numScenes=length( sub_HCI_dir );

     %% testing
     PSNR = zeros(numScenes, 1);
     SSIM = zeros(numScenes, 1);
     PSNR_var = zeros(numScenes, 1);
     SSIM_var = zeros(numScenes, 1);
     runTime=[];
     %% image size
     h=512;
     w=512;
     numImgsX = 7;
     numImgsY = 7;
    
 
    %% testing
    %% Metric : Y layer (PSNR, SSIM, PSNR_var, SSIM_var) RGB (PSNR, SSIM, PSNR_var, SSIM_var)
%     Metric = zeros(num_img, 8);
    
     for i = 1:numScenes
                
        if( isequal( sub_HCI_dir( i ).name, '.' )||...
              isequal( sub_HCI_dir( i ).name, '..')||...
             ~sub_HCI_dir( i ).isdir)               % 如果不是目录则跳过
             continue;
        end 
         
        img_name = sub_HCI_dir( i ).name;
        fprintf('Process Test Set %d/%d: %s\n', i, numScenes, img_name);

        if(saveImg)
            %mkdir(strcat(['Save_Img/' opts.model_name '/'  img_name]))
            mkdir(strcat(['Save_Img/Layer' num2str(depth) '/'  img_name]))
        end
        
        % Load HR image
        sceneFolder = opts.test_HCI_dir;
        scenePaths = fullfile( sceneFolder, img_name);
        hci_img = dir(fullfile(scenePaths, '/', '*.png'));
        
        fullLF = zeros(h, w, 3, numImgsY, numImgsX);
        inputLF = zeros(h, w, 3, numImgsY, numImgsX);
        
        for ns =1: length(hci_img)
           path_RGB = strcat(scenePaths, '/', hci_img(ns).name);
           im_RGB =im2double(imread(path_RGB)); %%将读取
           
           frame=ns-1;
           ay=floor(frame/7);
           temp=7*(ay);
           ax=frame-temp+1;
           ay=ay+1;
           
    
           fullLF(:, :, :, ay, ax)=im_RGB;  
           
           im_RGB = rgb2ycbcr(im_RGB);
           

           
           inputLF(:, :, :, ay, ax)=im_RGB;  
           
        end
        
        
        
        fullLF = reshape(fullLF, [w, h, 3, numImgsX*numImgsY]);
        inputLF = reshape(inputLF, [w, h, 3, numImgsX*numImgsY]);
       
        inputLF_Y = inputLF(:,:,1,:);
        [h,w,c,n] = size(inputLF_Y);
        

        
        
        %% Separate into sub-lightfields depending on len
        input_left = inputLF_Y(:,1:len+crop,:,:);
        inTensor = {};
        slice_data = floor(( w - len - crop )/ len );
        for sl = 1:slice_data
            inTensor{end+1} = inputLF_Y(:,sl*len+1-crop:(sl+1)*len+crop,:,:);
        end
        input_right = inputLF_Y(:,end-len+1-crop:end,:,:);
        
        %% Super-resolves sub-lightfields independently
        img_HR = zeros([h,w,40]);
        img_HR(crop+1:end-crop,crop+1:len,:) = Angular_SR(input_left, net, opts);
        for sl = 1:slice_data
            img_HR(crop+1:end-crop,1+sl*len:(sl+1)*len,:) = Angular_SR(inTensor{sl}, net, opts);
        end
        img_HR(crop+1:end-crop,end-len+1:end-crop,:) = Angular_SR(input_right, net, opts);
      
        img_HR = reshape(img_HR, [h,w,1,40]);
        
        
        %% Upsample with Bicubic Interploation in Angular Dimension
        %% First Part 
        inLF_1 = inputLF(:,:,:,[1,4,22,25]);
        inLF_1 = permute(inLF_1, [4,3,1,2]);
        inLF_1 = reshape(inLF_1, [4,3,h*w]);
        inLF_1 = reshape(inLF_1, [2,2,3,h*w]);
        inLF_1 = imresize(inLF_1, 2, 'bilinear');
        inLF_1 = reshape(inLF_1, [16,3,h*w]);
        inLF_1 = reshape(inLF_1, [16,3,h,w]);
        inLF_1 = permute(inLF_1, [3,4,2,1]);
        
        %disp(size(inLF_1));
        
        %%Second part
        inLF_2 = inputLF(:,:,:,[4,7,25,28]);
        inLF_2 = permute(inLF_2, [4,3,1,2]);
        inLF_2 = reshape(inLF_2, [4,3,h*w]);
        inLF_2 = reshape(inLF_2, [2,2,3,h*w]);
        inLF_2 = imresize(inLF_2, 2, 'bilinear');
        inLF_2 = reshape(inLF_2, [16,3,h*w]);
        inLF_2 = reshape(inLF_2, [16,3,h,w]);
        inLF_2 = permute(inLF_2, [3,4,2,1]);
        
        
        %%Third part
        inLF_3 = inputLF(:,:,:,[22,25,43,46]);
        inLF_3 = permute(inLF_3, [4,3,1,2]);
        inLF_3 = reshape(inLF_3, [4,3,h*w]);
        inLF_3 = reshape(inLF_3, [2,2,3,h*w]);
        inLF_3 = imresize(inLF_3, 2, 'bilinear');
        inLF_3 = reshape(inLF_3, [16,3,h*w]);
        inLF_3 = reshape(inLF_3, [16,3,h,w]);
        inLF_3 = permute(inLF_3, [3,4,2,1]);
        
        %%Fourth part
        inLF_4 = inputLF(:,:,:,[25,28,46,49]);
        inLF_4 = permute(inLF_4, [4,3,1,2]);
        inLF_4 = reshape(inLF_4, [4,3,h*w]);
        inLF_4 = reshape(inLF_4, [2,2,3,h*w]);
        inLF_4 = imresize(inLF_4, 2, 'bilinear');
        inLF_4 = reshape(inLF_4, [16,3,h*w]);
        inLF_4 = reshape(inLF_4, [16,3,h,w]);
        inLF_4 = permute(inLF_4, [3,4,2,1]);
        
        
        %% Combine
        % Fourth
%         inLF = inLF_4;       
%         inLF_W = inLF(:,:,:,[2:3,5:12,14:15]);       
%         imgHR = img_HR(:,:,[21:22,26:29,33:36,39:40]);
        
        % First
%         inLF = inLF_1;       
%         inLF_W = inLF(:,:,:,[2:3,5:12,14:15]);       
%         imgHR = img_HR(:,:,[1:2,5:8,12:15,19:20]);
        
        % Second
%         inLF = inLF_2;       
%         inLF_W = inLF(:,:,:,[2:3,5:12,14:15]);       
%         imgHR = img_HR(:,:,[3:4,8:11,15:18,21:22]);
        
        
        % THird
        inLF = inLF_3;       
        inLF_W = inLF(:,:,:,[2:3,5:12,14:15]);       
        imgHR = img_HR(:,:,[19:20,23:26,30:33,37:38]);
        
        
        inLF_W(:,:,1,:) = imgHR;
        img_HR = inLF_W;   
        
        for view = 1:16
          
          RGB_HR = ycbcr2rgb(img_HR(:,:,:,view));
          
          %RGB_HR = AdjustTone(RGB_HR);
          RGB_HR = shave_bd(RGB_HR, 22);
          imwrite(RGB_HR, strcat(['Save_Img/Layer' num2str(depth) '/' img_name '/' int2str(view), '.png']))
            
            
        end
        
        
        
        

%         %% Upsample with Bicubic Interploation in Angular Dimension
%         inLF = inputLF(:,:,:,[1,8,57,64]);
%         inLF = permute(inLF, [4,3,1,2]);
%         inLF = reshape(inLF, [4,3,h*w]);
%         inLF = reshape(inLF, [2,2,3,h*w]);
%         inLF = imresize(inLF, 4, 'bilinear');
%         inLF = reshape(inLF, [64,3,h*w]);
%         inLF = reshape(inLF, [64,3,h,w]);
%         inLF = permute(inLF, [3,4,2,1]);
%  
%         %% Combine
%         inLF = inLF(:,:,:,[2:7,9:56,58:63]);
%         inLF(:,:,1,:) = img_HR;
%         img_HR = inLF;

%         %% Evaluation
%         [f_w, f_h, f_c, f_a] = size(img_HR);
%         psnr_score = []; 
%         ssim_score = []; 
%         psnr_rgb_score = []; 
%         ssim_rgb_score = [];
% 
%         inputLF = inputLF(:,:,:,[2:7,9:56,58:63]);
%         fullLF = fullLF(:,:,:,[2:7,9:56,58:63]);
%        
%         for view = 1:60
%             
%             % Get a particular SAI for Y Layer
%             Y_HR = img_HR(:,:,1,view);
%             Y_GT = inputLF(:,:,1,view);
% 
%             % Get a particular SAI for RGB
%             RGB_HR = ycbcr2rgb(img_HR(:,:,:,view));
%             RGB_GT = fullLF(:,:,:,view);
% 
%             % Quantise pixels for the estimated SAI
%             Y_HR = im2double(im2uint8(Y_HR));   
%             RGB_HR = im2double(im2uint8(RGB_HR));   
% 
%             % Crop boundary
%             Y_HR = shave_bd(Y_HR, 22);
%             Y_GT = shave_bd(Y_GT, 22); 
%             RGB_HR = shave_bd(RGB_HR, 22);
%             RGB_GT = shave_bd(RGB_GT, 22);       
% 
% %             RGB_HR = AdjustTone(RGB_HR);
% %             RGB_GT = AdjustTone(RGB_GT);
%             
%             if(saveImg)
%                 imwrite(RGB_HR, strcat(['Save_Img/Layer' num2str(depth) '/' img_name '/' int2str(view), '.png']))
%             end
% 
%             % Evaluate
%             psnr_score(end+1) = psnr(Y_HR, Y_GT);
%             ssim_score(end+1) = ssim(Y_HR, Y_GT);
%             psnr_rgb_score(end+1) = (psnr(RGB_HR(:,:,1), RGB_GT(:,:,1))+psnr(RGB_HR(:,:,2), RGB_GT(:,:,2))+psnr(RGB_HR(:,:,3), RGB_GT(:,:,3)))/3;
%             ssim_rgb_score(end+1) = (ssim(RGB_HR(:,:,1), RGB_GT(:,:,1))+ssim(RGB_HR(:,:,2), RGB_GT(:,:,2))+ssim(RGB_HR(:,:,3), RGB_GT(:,:,3)))/3;
% 
%         end
% 
%         PSNR(i) = mean(psnr_score);
%         SSIM(i) = mean(ssim_score);
%         PSNR_var(i) = var(psnr_score);
%         SSIM_var(i) = var(ssim_score);
%         
%         disp(PSNR(i));
%         disp(SSIM(i));
        
        
        % average
%         Metric(i, :) = [mean(psnr_score), mean(ssim_score), var(psnr_score), var(ssim_score), mean(psnr_rgb_score), mean(ssim_rgb_score), var(psnr_rgb_score), var(ssim_rgb_score)];
%         
%         disp(Metric(i,1));
%         disp(Metric(i,2));       

    end
        
%     PSNR_Y_mean = mean(Metric(:,1));
%     SSIM_Y_mean = mean(Metric(:,2));
%     PSNR_RGB_mean = mean(Metric(:,5));
%     SSIM_RGB_mean = mean(Metric(:,6));
%  
%     % write result to csv file
%     %csvwrite(strcat(['Result_CSV/', 'Layer' num2str(depth) '_Result.csv']),Metric);
%     
%     fprintf('Average PSNR Y Layer = %f\n', PSNR_Y_mean);
%     fprintf('Average SSIM Y Layer = %f\n', SSIM_Y_mean);
%     fprintf('Average PSNR RGB = %f\n', PSNR_RGB_mean);
%     fprintf('Average SSIM RGB = %f\n', SSIM_RGB_mean);
