function cl_percentage = HDC_CIFAR10(image_row_size,image_column_size,D,images_train1,images_test1,P_hypervector...
    ,intensity_vector,labels_train,labels_test,total_training_images,total_test_images)
%coder.extrinsic('meh_train_mex','meh_test_mex');
%trainDB_size = 1;
%total_training_images = 1000;

numberOfClasses = 10;
accuracy = 0;
%xored_Images = zeros(1, 28, 28, 1024);
%xored_Images = zeros(28, 28, 1024);
%bundled = zeros(1, 1024);
%reshaped_P_hv = zeros(1, 1024);
%shaped_images = zeros(1, 28, 28);
%shaped_images = zeros(28, 28);

%Status bar
%WaitMessage = parfor_wait(total_training_images, 'Waitbar', true);

%TRAINING STARTS

%cumulative_class_hypervector = zeros(10,1024);

cumulative_class_hypervector0 = zeros(1,D);
cumulative_class_hypervector1 = zeros(1,D);
cumulative_class_hypervector2 = zeros(1,D);
cumulative_class_hypervector3 = zeros(1,D);
cumulative_class_hypervector4 = zeros(1,D);
cumulative_class_hypervector5 = zeros(1,D);
cumulative_class_hypervector6 = zeros(1,D);
cumulative_class_hypervector7 = zeros(1,D);
cumulative_class_hypervector8 = zeros(1,D);
cumulative_class_hypervector9 = zeros(1,D);



%**************************Mehran_New**************************************

for k = 2:log2(D)
    vd(:,k-1) = vdcorput((D-1),k);
end

alpha = pi;
beta = exp(1);
weyl(:,1) = mod((1:D)*alpha, 1);
weyl(:,2) = mod((1:D)*beta, 1);

ws16k1 = load('sobol_bl_optimized_16k_91.mat','aa8k1');
aa16k = ws16k1.aa8k1;
%aa16k = aa16k(1:image_row_size*image_column_size);


sobol_seq1 = net(sobolset(1111), D);
sobol_seq_new = sobol_seq1(:,aa16k);

sobol_seq_new = sobol_seq_new(:,1:313);
sobol_seq_new = [sobol_seq_new,vd,weyl];
sobol_seq_new = sobol_seq_new(:,randperm(image_row_size*image_column_size));




tic
parfor TRAIN_IMAGE_INDEX = 1:total_training_images %50 will be total db size

    %--------------------------------------------------------------------------
    %Status bar odds
    % Update waitbar and message
    %WaitMessage.Send;
    %pause(0.002);
    %--------------------------------------------------------------------------

    %--------------------------------------------------------------------------
    %--------------------------------------------------------------------------
    %Generating level hypervectors L
    %feature size is image_row * image_column

    %Pre-process dataset
    %shaped_images = zeros(1, 28, 28);

    % for i = 1:1:trainDB_size
    %     shaped_images(i,:,:) = reshape(images_train(:, i), [1,28,28]);
    % end
    %k = 0;
    %{
    for ii = 0:1:27
        shaped_images(:,ii+1) = images_train1((ii*28)+1:(ii*28)+28, TRAIN_IMAGE_INDEX)';
        %k = k + 28;
    end
    %}

    shaped_images = shapedim18_mex(images_train1(:,TRAIN_IMAGE_INDEX));


    %shaped_images = uint8(shaped_images);

    %shaped_images(1,:,:) = reshape(images_train1(:, TRAIN_IMAGE_INDEX), [1,28,28]);

    %single image shaping ---> reshape(shaped_images(1,:,:), [28,28])

    %shaped_images
    % 3-dimensional vector (currently) ---> image index * row * column

    %Image quantization if needed (be aware to update low & high intensities & M value, even maybe the D value)
    %shaped_images = floor(shaped_images ./ 32);

    %Level hypervectors
    %Allocate mem.
    %vectorized_Images = zeros(trainDB_size, image_row_size, image_column_size, D);


    %--------------------------------------------------------------------------
    %XOR (i.e. multiplication)
    %----------------------------BINDING---------------------------------------
    %Allocate mem.
    %xored_Images = zeros(1, 28, 28, 1024);
    %temp = zeros(1, 1, 1, 1024);
    %reshaped_P_hv = zeros(1, 1024);
    %k = 1;

    %for item_image=1:1:trainDB_size %iterating over training images
        %{
        for i=1:1:image_row_size
            %shaped_images(1,image_row_size,:) = images_train(k:k+27, TRAIN_IMAGE_INDEX);
            for j = 1:1:image_column_size
                %for meh = 1:D
                 %   reshaped_P_hv(item_image, meh) = P_hypervector(i, j, meh);
                %end
                reshaped_P_hv(1,:) = P_hypervector(i,j,:);
                %temp = reshaped_P_hv(item_image, :) .* intensity_vector(shaped_images(item_image,i,j)+1,:);
                %temp = reshape(P_hypervector(i,j,:), [1,D]) .* intensity_vector(shaped_images(item_image,i,j)+1,:);              
                %xored_Images(item_image, i, j, :) = reshape(temp, [1,1,1,D]);                
                %xored_Images(item_image, i, j, :) = reshaped_P_hv(item_image,:) .* intensity_vector(shaped_images(item_image,i,j)+1,:);
                xored_Images(i, j, :) = reshaped_P_hv(1,:) .* intensity_vector(shaped_images(i,j)+1,:);
                
            end
        end
        %}
    %*********************************************************
%         rscale_imgtr = rescale(images_train1(:,TRAIN_IMAGE_INDEX),0,1);
%         %rscale_imgtr = reshape(rscale_imgtr,image_row_size,image_column_size);
%         PP_sobol = ones(D,image_row_size*image_column_size);
%         for i = 1:image_row_size*image_column_size
%             %for j = 1:image_column_size
%                 for z = 1:D
%                     if rscale_imgtr(i) <= sobol_seq_new(z,i)
%                         PP_sobol(z,i) = -1;
%                     end            
%                 end    
%             %end
%         end
%         PP_sobol = PP_sobol';
%         xored_Images = reshape(PP_sobol,image_row_size,image_column_size,D);


        xored_Images = binding8k_18_mex(image_row_size,image_column_size,D,P_hypervector,intensity_vector,shaped_images);
        %xored_Images = P_hypervector;

%         xored_Images = zeros(image_row_size,image_column_size, D);
%         reshaped_P_hv = zeros(1, D);
%         for i=1:1:image_row_size
%             %shaped_images(1,image_row_size,:) = images_train(k:k+27, TRAIN_IMAGE_INDEX);
%             for j = 1:1:image_column_size
%                 %for meh = 1:D
%                  %   reshaped_P_hv(item_image, meh) = P_hypervector(i, j, meh);
%                 %end
%                 reshaped_P_hv(1,:) = P_hypervector(i,j,:);
%                 xored_Images(i, j, :) = reshaped_P_hv(1,:) .* intensity_vector(shaped_images(i,j)+1,:);
%                 
%             end
%         end


        %xored_Images = output_sobol(images_train1(:,TRAIN_IMAGE_INDEX),TRAIN_IMAGE_INDEX);
        %xored_Images = output_hadamard(shaped_images,V_hd);
        %xored_Images = reshape(xx(TRAIN_IMAGE_INDEX,:,:),image_row_size,image_column_size,D);

%         if TRAIN_IMAGE_INDEX == 1
%             xored_Images = output_sobol(images_train1(:,TRAIN_IMAGE_INDEX),TRAIN_IMAGE_INDEX);
%         else
%             xored_Images = output_sobol(images_train1(:,TRAIN_IMAGE_INDEX),dd1(TRAIN_IMAGE_INDEX));
%         end
    %end
    
    %xor_Control = reshape(xored_Images(1,1,1,:), [1,D]);
    %--------------------------------------------------------------------------
    %xored_Images = meh_train_mex(image_row_size,image_column_size,D,images_train,P_hypervector,intensity_vector,TRAIN_IMAGE_INDEX);

    %--------------------------------------------------------------------------
    %ADD
    %----------------------------BUNDLING--------------------------------------
    %Allocate mem.
    %{
    reshaped_xored_Images = zeros(1, 1024);

    %for item_image=1:1:trainDB_size %iterating over training images
        for i=1:1:image_row_size
            for j = 1:1:image_column_size
                %for meh = 1:1:D
                 %   reshaped_xored_Images(item_image, meh) = xored_Images(item_image, i, j, meh);
                %end
                %reshaped_xored_Images(item_image,:) = xored_Images(item_image,i,j,:);
                reshaped_xored_Images(1,:) = xored_Images(i,j,:);
                %bundled(item_image, :) = bundled(item_image, :) + reshaped_xored_Images(item_image, :);
                bundled(1, :) = bundled(1, :) + reshaped_xored_Images(1, :);
            end
        end
    %}
    bundled = bundling8k_18_mex(image_row_size,image_column_size,D,xored_Images);
    %end
    %--------------------------------------------------------------------------


    %--------------------------------------------------------------------------
    %SIGN
    %----------------------------BINARIZING------------------------------------
    %Allocate mem.
    bundled_signed = sign(bundled);
    %bundled_signed = bundled;
    %Exception handling for `0` values
    %{
    for z = 1:1:D
        if bundled_signed(z) == 0 %0 is 1 for us
            bundled_signed(z) = 1;
        end
    end
    %}
    %--------------------------------------------------------------------------
    %********************************************************
    %Mehran-vectorizing
    bb = bundled_signed;
    bb(bundled_signed == 0) = 1;
    bundled_signed = bb;

    %********************************************************
    %--------------------------------------------------------------------------
    %-------------------------CLASS HYPERVECTOR--------------------------------
    %Allocate mem.
    %cumulative_class_hypervector = zeros(numberOfClasses,D); %temporary assignment
    
    %EITHER----------------------------------------------------------------
    %The following was a better choice for cumulative_class_hypervector, but parfor gets angry :(
    %cumulative_class_hypervector(labels_train(TRAIN_IMAGE_INDEX)+1,:) = cumulative_class_hypervector(labels_train(TRAIN_IMAGE_INDEX)+1,:) + bundled_signed;
    %EITHER----------------------------------------------------------------

    %OR--------------------------------------------------------------------
    if labels_train(TRAIN_IMAGE_INDEX) == 0
        cumulative_class_hypervector0 = cumulative_class_hypervector0 + bundled_signed;
    end
    if labels_train(TRAIN_IMAGE_INDEX) == 1
        cumulative_class_hypervector1 = cumulative_class_hypervector1 + bundled_signed;
    end
    if labels_train(TRAIN_IMAGE_INDEX) == 2
        cumulative_class_hypervector2 = cumulative_class_hypervector2 + bundled_signed;
    end
    if labels_train(TRAIN_IMAGE_INDEX) == 3
        cumulative_class_hypervector3 = cumulative_class_hypervector3 + bundled_signed;
    end
    if labels_train(TRAIN_IMAGE_INDEX) == 4
        cumulative_class_hypervector4 = cumulative_class_hypervector4 + bundled_signed;
    end
    if labels_train(TRAIN_IMAGE_INDEX) == 5
        cumulative_class_hypervector5 = cumulative_class_hypervector5 + bundled_signed;
    end
    if labels_train(TRAIN_IMAGE_INDEX) == 6
        cumulative_class_hypervector6 = cumulative_class_hypervector6 + bundled_signed;
    end
    if labels_train(TRAIN_IMAGE_INDEX) == 7
        cumulative_class_hypervector7 = cumulative_class_hypervector7 + bundled_signed;
    end
    if labels_train(TRAIN_IMAGE_INDEX) == 8
        cumulative_class_hypervector8 = cumulative_class_hypervector8 + bundled_signed;
    end
    if labels_train(TRAIN_IMAGE_INDEX) == 9
        cumulative_class_hypervector9 = cumulative_class_hypervector9 + bundled_signed;
    end
    %OR--------------------------------------------------------------------

end %end of training iteration
toc
%WaitMessage.Destroy %close status bar

%OR--------------------------------------------------------------------
cumulative_class_hypervector = cat(1, cumulative_class_hypervector0, cumulative_class_hypervector1, cumulative_class_hypervector2, ...
    cumulative_class_hypervector3, cumulative_class_hypervector4, cumulative_class_hypervector5, cumulative_class_hypervector6, ...
    cumulative_class_hypervector7, cumulative_class_hypervector8, cumulative_class_hypervector9);
%OR--------------------------------------------------------------------


%BINARY--------------------------------------------------------------------
%---------------------------CLASS HYPERVECTOR SIGN-------------------------
signed_class_hypervector = (cumulative_class_hypervector);
%signed_class_hypervector = sign(cumulative_class_hypervector);

%{
%zero correction
zero_index = find(~signed_class_hypervector);
zero_index_size = size(zero_index);
for z = 1:1:zero_index_size
    signed_class_hypervector(zero_index(z)) = 1;
end
%}
%*****************************************************************
%	Mehran

% cc = signed_class_hypervector;
% cc(signed_class_hypervector == 0) = 1;
% signed_class_hypervector = cc;

%*****************************************************************

%---------------------------CLASS HYPERVECTOR SIGN-------------------------


%NON-BINARY--------------------------------------------------------------------
%---------------------------CLASS HYPERVECTOR SIGN-------------------------
%signed_class_hypervector = cumulative_class_hypervector;
%---------------------------CLASS HYPERVECTOR SIGN-------------------------


%          signed_class_hypervector is the model for inference

%                              END of TRAING
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%##########################################################################
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------


%%
%                                INFERENCE

%TESTING STARTS
%Status bar
%h = waitbar(0,'TESTING');


%total_test_images = 100;
tic
parfor TESTING_IMAGE_INDEX = 1:total_test_images %50 will be total db size
    
    %--------------------------------------------------------------------------
    %Status bar odds
    % Update waitbar and message
    %waitbar(TESTING_IMAGE_INDEX/total_test_images,h)
    %--------------------------------------------------------------------------

    %----------------------------------------------------------------------
    %----------------------------------------------------------------------
    %Generating level hypervectors L
    %feature size is image_row * image_column

    %Pre-process dataset
    %shaped_images = zeros(1, 28, 28);

    % for i = 1:1:trainDB_size
    %     shaped_images(i,:,:) = reshape(images_train(:, i), [1,28,28]);
    % end
    %{
    for ii = 0:1:27
        shaped_images(:,ii+1) = images_test1((ii*28)+1:(ii*28)+28, TESTING_IMAGE_INDEX)';
    end
    %}
    shaped_images = shapedim18_mex(images_test1(:,TESTING_IMAGE_INDEX));
    %shaped_images(1,:,:) = reshape(images_test(:, TESTING_IMAGE_INDEX), [1,28,28]);


    %single image shaping ---> reshape(shaped_images(1,:,:), [28,28])

    %shaped_images
    % 3-dimensional vector (currently) ---> image index * row * column


    %Image quantization if needed (be aware to update low & high intensities & M value, even maybe the D value)
    %shaped_images = floor(shaped_images ./ 32);

    %Level hypervectors
    %Allocate mem.
    %vectorized_Images = zeros(trainDB_size, image_row_size, image_column_size, D);


    %intensity has M * D size, where M is the quantized intervals (total different pixel values --> 0...255 etc.)
    % intensity_vector(1,:) --> 1 1 1 ... 1
    % and
    % intensity_vector(M,:) --> 0 0 0 ... 0 for 1-bitflip_count at each level

    %----------------------------------------------------------------------
    %XOR (i.e. multiplication)
    %----------------------------BINDING-----------------------------------
    %Allocate mem.
    %xored_Images = zeros(1, 28, 28, 1024);
    %temp = zeros(1, 1024);
    %reshaped_P_hv = zeros(1, 1024);

    %for item_image=1:1:trainDB_size %iterating over training images
    %{
        for i=1:1:image_row_size
            for j = 1:1:image_column_size
                %for meh = 1:D
                 %   reshaped_P_hv(item_image, meh) = P_hypervector(i, j, meh);
                %end
                reshaped_P_hv(1, :) = P_hypervector(i, j, :);
                %xored_Images(item_image, i, j, :) = reshaped_P_hv(item_image, :) .* intensity_vector(shaped_images(item_image,i,j)+1,:);
                xored_Images(i, j, :) = reshaped_P_hv(1, :) .* intensity_vector(shaped_images(i,j)+1,:);
                %temp = reshape(P_hypervector(i,j,:), [1,D]) .* intensity_vector(shaped_images(item_image,i,j)+1,:);
                %xored_Images(item_image, i, j, :) = reshape(temp, [1,1,1,D]);
            end
        end
    %}
    %end
    %******************************************************
%         rscale_imgts = rescale(images_test1(:,TESTING_IMAGE_INDEX),0,1);
%         %rscale_imgtr = reshape(rscale_imgtr,image_row_size,image_column_size);
%         PP_sobol1 = ones(D,image_row_size*image_column_size);
%         for i = 1:image_row_size*image_column_size
%             %for j = 1:image_column_size
%                 for z = 1:D
%                     if rscale_imgts(i) <= sobol_seq_new(z,i)
%                         PP_sobol1(z,i) = -1;
%                     end            
%                 end    
%             %end
%         end
%         PP_sobol1 = PP_sobol1';
%         xored_Images = reshape(PP_sobol1,image_row_size,image_column_size,D);



    xored_Images = binding8k_18_mex(image_row_size,image_column_size,D,P_hypervector,intensity_vector,shaped_images);
    %xored_Images = P_hypervector;

%     xored_Images = zeros(image_row_size,image_column_size, D);
%     reshaped_P_hv = zeros(1, D);
%     for i=1:1:image_row_size
%             %shaped_images(1,image_row_size,:) = images_train(k:k+27, TRAIN_IMAGE_INDEX);
%         for j = 1:1:image_column_size
%                 %for meh = 1:D
%                  %   reshaped_P_hv(item_image, meh) = P_hypervector(i, j, meh);
%                 %end
%                 reshaped_P_hv(1,:) = P_hypervector(i,j,:);
%                 xored_Images(i, j, :) = reshaped_P_hv(1,:) .* intensity_vector(shaped_images(i,j)+1,:);
%                 
%         end
%     end



    %xored_Images = output_sobol(images_test1(:,TESTING_IMAGE_INDEX),TESTING_IMAGE_INDEX);
    %xored_Images = output_hadamard(shaped_images,V_hd);
    %xored_Images = reshape(yy(TESTING_IMAGE_INDEX,:,:),image_row_size,image_column_size,D);

%     if TESTING_IMAGE_INDEX == 1
%         xored_Images = output_sobol(images_test1(:,TESTING_IMAGE_INDEX),TESTING_IMAGE_INDEX);
%     else
%         xored_Images = output_sobol(images_test1(:,TESTING_IMAGE_INDEX),dd1(TESTING_IMAGE_INDEX));
%     end
    
    %xor_Control = reshape(xored_Images(1,1,1,:), [1,D]);
    %--------------------------------------------------------------------------
    %**************************************************************************
    %Mehran
    %xored_Images = meh_test_mex(image_row_size,image_column_size,D,images_test,P_hypervector,intensity_vector,TESTING_IMAGE_INDEX);

    %**************************************************************************
    %--------------------------------------------------------------------------
    %ADD
    %----------------------------BUNDLING--------------------------------------
    %Allocate mem.
    %{
    %bundled = zeros(1, 1024);
    reshaped_xored_Images = zeros(1, 1024);
    %for item_image=1:1:trainDB_size %iterating over training images
        for i=1:1:image_row_size
            for j = 1:1:image_column_size
                %for meh = 1:1:D
                 %   reshaped_xored_Images(item_image, meh) = xored_Images(item_image, i, j, meh);
                %end
                reshaped_xored_Images(1, :) = xored_Images(i, j, :);
                %bundled(item_image, :) = bundled(item_image, :) + reshaped_xored_Images(item_image, :);
                bundled(1, :) = bundled(1, :) + reshaped_xored_Images(1, :);
            end
        end
    %end
    %}
    bundled = bundling8k_18_mex(image_row_size,image_column_size,D,xored_Images);
    %----------------------------------------------------------------------


    %----------------------------------------------------------------------
    %SIGN
    %----------------------------BINARIZING--------------------------------
    %Allocate mem.
    bundled_signed = sign(bundled);
    %bundled_signed = bundled;
    %Exception handling for `0` values
    %{
    for z = 1:1:D
        if bundled_signed(z) == 0 %0 is 1 for us
            bundled_signed(z) = 1;
        end
    end
    %}
    %----------------------------------------------------------------------
    %**********************************************************************
    %Mehran
    dd = bundled_signed;
    dd(bundled_signed == 0) = 1;
    bundled_signed = dd;

    %**********************************************************************
    %CLASSIFICATION
    cosAngle = zeros(1, 10);
    dtprod = zeros(1, D);
    nrm_schv = zeros(1, 10);

    nrm_bundled = sqrt(sum(bundled_signed .^ 2));
    
    for classes = 1:1:numberOfClasses
        dtprod(classes) = sum(bundled_signed .* signed_class_hypervector(classes,:));
        nrm_schv(classes) = sqrt(sum(signed_class_hypervector(classes,:) .^ 2));
        cosAngle(classes) = dtprod(classes)/(nrm_bundled * nrm_schv(classes));
        %cosAngle(classes) = dot(bundled_signed(1,:), signed_class_hypervector(classes,:))/(norm(bundled_signed(1,:))*norm(signed_class_hypervector(classes, :)));
    end
    
    

    [~, position] = max(cosAngle);

    if position == (labels_test(TESTING_IMAGE_INDEX)+1)
        accuracy = accuracy + 1;
    end

end %end of testing for 
%delete(h);
toc
cl_percentage = (accuracy * 100) / total_test_images;