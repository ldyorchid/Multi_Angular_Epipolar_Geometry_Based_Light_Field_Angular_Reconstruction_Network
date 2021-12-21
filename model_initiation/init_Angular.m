function net = init_Angular(opts)
% -------------------------------------------------------------------------
%   Description:
%       create initial LapSRN model
%       Modified from the code produced by the authors in the citation below
%
%   Input:
%       - opts  : options generated from init_opts()
%
%   Output:
%       - net   : dagnn model
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

    %% parameters
    rng('default');
    rng(0) ;
    
    % filter width
    f       = opts.conv_f;    % f equals to 3
    % number of filters set to 64
    n       = opts.conv_n;    % n equals to 49
    % angular resolution should be 49 

 
    %pad     = floor(f/2);     
    
    depth   = opts.depth; % depth is the conv layers number  
    %patch_size = opts.patch_size; 
    level = 1;
  
    net = dagnn.DagNN;


%% **************************************************************************
    %%  Feature extraction of multiple branches structure 
%% ************************************************************************** 
    sigma   = opts.init_sigma;
    filters = sigma * randn(f, f, f, 1, 49, 'single');
    biases  = zeros(1, 49, 'single');
    pad=[1,1,1,1,1,1];  
    stride = [1,1,1];
   
    for i=1:4
       p=(i-1)*45 ;
       % conv: 5D Filter
	   inputs  = { sprintf('LR_Angle_%d', p)};
	   outputs = { sprintf('conv5d_LR_Angle_%d', p)};
	   params  = { sprintf('conv5d_LR_Angle_%d_f', p), ...
		           sprintf('conv5d_LR_Angle_%d_b', p)};

	   net.addLayer(outputs{1}, ...
		      dagnn.Conv5D('size', size(filters), ...
			               'pad', pad, ...
			               'stride', stride), ...
		                   inputs, outputs, params);

	    idx = net.getParamIndex(params{1});
	    net.params(idx).value         = filters;
	    net.params(idx).learningRate  = 1;
	    net.params(idx).weightDecay   = 1;

	    idx = net.getParamIndex(params{2});
	    net.params(idx).value         = biases;
	    net.params(idx).learningRate  = 0.1;
	    net.params(idx).weightDecay   = 1;
       
        next_input=outputs{1};

    % relu

        inputs  = { next_input };
        outputs = { sprintf('relu_LR_Angle_%d', p) };

        net.addLayer(outputs{1}, ...
                     dagnn.ReLU('leak', 0.2), ...
                     inputs, outputs);  
         
    end


%% **********************************************************************************
    %% Sum of the obtained multiple branches features
%% **********************************************************************************

    inputs  = { sprintf('relu_LR_Angle_%d', 0), ...
                sprintf('relu_LR_Angle_%d', 45)};
    outputs = { 'sum_Angle_0_45' };
                net.addLayer(outputs{1}, ...
                dagnn.Sum(), ...
                inputs, outputs);
        
    next_input = outputs{1}; 
    
    inputs  = { next_input, ...
                sprintf('relu_LR_Angle_%d', 90)};
    outputs = { 'sum_Angle_0_45_90' };
                net.addLayer(outputs{1}, ...
                dagnn.Sum(), ...
                inputs, outputs);

    next_input = outputs{1}; 
    
    inputs  = { next_input, ...
                sprintf('relu_LR_Angle_%d', 135)};
    outputs = { sprintf('conv5d_Angle_%d', 0)};   %'sum_Angle_0_45_90_135'
                net.addLayer(outputs{1}, ...
                dagnn.Sum(), ...
                inputs, outputs);

    next_input = outputs{1};   % Size of the next_input is 64x64x3x49


%% ***********************************************************************
     %% deep conv layers (f x f x 1 x n)  
     %% The feature extraction is end, then the convs are executed
 %% ***********************************************************************   
    sigma   = sqrt( 2 / (f * f * f * 1 * n) );
         
    for s = 1:depth
        
        % First conv: 5D Filter    
        filters = sigma * randn(f, f, f, n, n, 'single');
        biases  = zeros(1, n, 'single');
        pad=[1,1,1,1,1,1];  
        stride = [1,1,1];   
   
        inputs  = { next_input};
        outputs = { sprintf('conv5d_MultiB_%d', s) };
        params  = { sprintf('conv5d_MultiB_f_%d', s), ...
                    sprintf('conv5d_MultiB_b_%d', s)};

        net.addLayer(outputs{1}, ...
                 dagnn.Conv5D('size', size(filters), ...
                            'pad', pad, ...
                            'stride', stride), ...
                 inputs, outputs, params);

        idx = net.getParamIndex(params{1});
        net.params(idx).value         = filters;
        net.params(idx).learningRate  = 1;
        net.params(idx).weightDecay   = 1;

        idx = net.getParamIndex(params{2});
        net.params(idx).value         = biases;
        net.params(idx).learningRate  = 0.1;
        net.params(idx).weightDecay   = 1;
      
        next_input = outputs{1}; 
      
        % ReLU
        inputs  = { next_input };
        outputs = { sprintf('conv5d_MultiB_relu_%d', s) };

        net.addLayer(outputs{1}, ...
                 dagnn.ReLU('leak', 0.2), ...
                 inputs, outputs);
                        
        next_input = outputs{1};
        
        % Second conv: 5D Filter 
        sigma   = sqrt( 2 / (f * f * f* n * n) );
        filters = sigma * randn(f, f, f, n, n, 'single');
        biases  = zeros(1, n, 'single');
        pad=[1,1,1,1,1,1];  
        stride = [1,1,1];   
   
        inputs  = { next_input};
        outputs = { sprintf('conv5d_MultiB_2_%d', s)  };
        params  = { sprintf('conv5d_MultiB_2_f_%d', s) , ...
                    sprintf('conv5d_MultiB_2_b_%d', s) };

        net.addLayer(outputs{1}, ...
                 dagnn.Conv5D('size', size(filters), ...
                            'pad', pad, ...
                            'stride', stride), ...
                 inputs, outputs, params);

        idx = net.getParamIndex(params{1});
        net.params(idx).value         = filters;
        net.params(idx).learningRate  = 1;
        net.params(idx).weightDecay   = 1;

        idx = net.getParamIndex(params{2});
        net.params(idx).value         = biases;
        net.params(idx).learningRate  = 0.1;
        net.params(idx).weightDecay   = 1;            
             
        next_input = outputs{1};     
             
        % add
        inputs  = { next_input , ...
                    sprintf('conv5d_Angle_%d', s-1)};
        outputs = { sprintf('conv5d_Angle_%d', s)};
        net.addLayer(outputs{1}, ...
            dagnn.Sum(), ...
            inputs, outputs);             
             
        
         next_input=outputs{1};
    end
    
 %the size of the next_input is 64x64x3x49


%% **************************************************************************
    %% Novel SAIs synthesis layer (f * f * 2 * n * n)
%% **************************************************************************   
   
    sigma   = sqrt(2 / (f * f * f * n * n));
    
    filters = sigma * randn(f, f, 2, n, 49, 'single');
    biases  = zeros(1, 49, 'single');
    pad=[1,1,1,1,0,0];  
    stride = [1,1,1];   
    % conv: 6D Filter
    inputs  = { next_input};
    outputs = { 'conv5d_Syn_1' };
    params  = { 'conv5d_f_Syn_1', ...
    	    'conv5d_b_Syn_1'};

    net.addLayer(outputs{1}, ...
	 dagnn.Conv5D('size', size(filters), ...
	            'pad', pad, ...
	            'stride', stride), ...
	 inputs, outputs, params);

    idx = net.getParamIndex(params{1});
    net.params(idx).value         = filters;
    net.params(idx).learningRate  = 1;
    net.params(idx).weightDecay   = 1;

    idx = net.getParamIndex(params{2});
    net.params(idx).value         = biases;
    net.params(idx).learningRate  = 0.1;
    net.params(idx).weightDecay   = 1;
    
% ReLU 
    inputs  = { 'conv5d_Syn_1'};
    outputs = { 'relu_syn_1'};
 
    net.addLayer(outputs{1}, ...
                  dagnn.ReLU('leak', 0.2), ...
                  inputs, outputs);     
    
 % *************conv ********************  
    sigma   = sqrt(2 / (f * f * 2 * n * n));
    
    filters = sigma * randn(f, f, 2, n, 40, 'single');
    biases  = zeros(1, 40, 'single');
    pad=[1,1,1,1,0,0];  
    stride = [1,1,1];   
    % conv: 6D Filter
    inputs  = { 'relu_syn_1'};
    outputs = { 'conv5d_Syn_2' };
    params  = { 'conv5d_f_Syn_2', ...
    	        'conv5d_b_Syn_2'};

    net.addLayer(outputs{1}, ...
	 dagnn.Conv5D('size', size(filters), ...
	            'pad', pad, ...
	            'stride', stride), ...
	 inputs, outputs, params);

    idx = net.getParamIndex(params{1});
    net.params(idx).value         = filters;
    net.params(idx).learningRate  = 1;
    net.params(idx).weightDecay   = 1;

    idx = net.getParamIndex(params{2});
    net.params(idx).value         = biases;
    net.params(idx).learningRate  = 0.1;
    net.params(idx).weightDecay   = 1;
        
    % the size of 'conv5d_Syn_2' is 64x64x1x40
      
    %next_input=outputs{1};         
%% ************************************************************************
   %%  Concat and Reshape, the output is of size 64x64x7x7
%% ************************************************************************     
    inputs  = { 'conv5d_Syn_2', 'LR' };
    outputs = { 'concat' };
    net.addLayer(outputs{1}, dagnn.Concat_Reshape(), inputs, outputs)

    next_input=outputs{1};  

%% ************************************************************************
   %%  Refinement Network
%% ************************************************************************     
% *************conv ********************  
    sigma   = sqrt(2 / (f * f * 2 * n * 40));
    
    filters = sigma * randn(f, f, f, 7, 49, 'single');
    biases  = zeros(1, 49, 'single');
    pad=[1,1,1,1,0,0];  
    stride = [1,1,2];   
    % conv: 6D Filter
    inputs  = { next_input};
    outputs = { 'conv5d_Refine_1' };
    params  = { 'conv5d_f_Refine_1', ...
    	    'conv5d_b_Refine_1'};

    net.addLayer(outputs{1}, ...
	 dagnn.Conv5D('size', size(filters), ...
	            'pad', pad, ...
	            'stride', stride), ...
	 inputs, outputs, params);

    idx = net.getParamIndex(params{1});
    net.params(idx).value         = filters;
    net.params(idx).learningRate  = 1;
    net.params(idx).weightDecay   = 1;

    idx = net.getParamIndex(params{2});
    net.params(idx).value         = biases;
    net.params(idx).learningRate  = 0.1;
    net.params(idx).weightDecay   = 1;
   
    
% ReLU 
    inputs  = { 'conv5d_Refine_1'};
    outputs = { 'relu_Refine_1'};
 
    net.addLayer(outputs{1}, ...
                  dagnn.ReLU('leak', 0.2), ...
                  inputs, outputs);  
   
         
    next_input=outputs{1};  % the size is 64x64x3x16
   % *************conv ********************  
    sigma   = sqrt(2 / (f * f * f * 49 * 7));
    
    filters = sigma * randn(f, f, 2, 49, n, 'single');
    biases  = zeros(1, n, 'single');
    pad=[1,1,1,1,0,0];  
    stride = [1,1,1];   
    % conv: 6D Filter
    inputs  = { next_input};
    outputs = { 'conv5d_Refine_2' };
    params  = { 'conv5d_f_Refine_2', ...
    	    'conv5d_b_Refine_2'};

    net.addLayer(outputs{1}, ...
	 dagnn.Conv5D('size', size(filters), ...
	            'pad', pad, ...
	            'stride', stride), ...
	 inputs, outputs, params);

    idx = net.getParamIndex(params{1});
    net.params(idx).value         = filters;
    net.params(idx).learningRate  = 1;
    net.params(idx).weightDecay   = 1;

    idx = net.getParamIndex(params{2});
    net.params(idx).value         = biases;
    net.params(idx).learningRate  = 0.1;
    net.params(idx).weightDecay   = 1;
              
              
 % ReLU 
    inputs  = { 'conv5d_Refine_2'};
    outputs = { 'relu_Refine_2'};
 
    net.addLayer(outputs{1}, ...
                  dagnn.ReLU('leak', 0.2), ...
                  inputs, outputs);               
              
%% ************************************************************************
   %%  Detail Recovery
%% ************************************************************************               
    % *************conv ********************  
    sigma   = sqrt(2 / (f * f * 2 * n * n));
    
    filters = sigma * randn(f, f, 2, n, 40, 'single');
    biases  = zeros(1, 40, 'single');
    pad=[1,1,1,1,0,0];  
    stride = [1,1,1];   
    % conv: 6D Filter
    inputs  = { 'relu_Refine_2'};
    outputs = { 'conv5d_Detail' };
    params  = { 'conv5d_f_Detail', ...
    	    'conv5d_b_Detail'};

    net.addLayer(outputs{1}, ...
	 dagnn.Conv5D('size', size(filters), ...
	            'pad', pad, ...
	            'stride', stride), ...
	 inputs, outputs, params);

    idx = net.getParamIndex(params{1});
    net.params(idx).value         = filters;
    net.params(idx).learningRate  = 1;
    net.params(idx).weightDecay   = 1;

    idx = net.getParamIndex(params{2});
    net.params(idx).value         = biases;
    net.params(idx).learningRate  = 0.1;
    net.params(idx).weightDecay   = 1;             
         
    next_input=outputs{1}; 
    
%% ************************************************************************
   %%   loss layer
%% ************************************************************************     
    for s = level : -1 : 1
        
        % add
        inputs  = { next_input, ...
                    'conv5d_Syn_2'};
        outputs = { 'output' };
        net.addLayer(outputs{1}, ...
            dagnn.Sum(), ...
            inputs, outputs);
        
        next_input = outputs{1}; 
             
        
        inputs  = { next_input, ...
                    'HR' };
        outputs = { sprintf('%s_loss', opts.loss) };
        
        net.addLayer(outputs{1}, ...
                 dagnn.vllab_dag_loss(...
                    'loss_type', opts.loss), ...
                 inputs, outputs);
             
                
    end   
             

end
