function cl_percentage = ECT_HDC_Meh_opt_tr_quantized_ts_noqu(image_row_size,image_column_size,D,images_train1,images_test1,P_hypervector...
    ,intensity_vector,labels_train,labels_test,total_training_images,total_test_images)
%coder.extrinsic('meh_train_mex','meh_test_mex');
%trainDB_size = 1;
%total_training_images = 1000;

%numberOfClasses = 10;
numberOfClasses = max(labels_test) + 1;
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

% ws_hdv = load("hadamard_img_tensor.mat");
% V_hd = ws_hdv.V_hd;
% ws_hdv = load("hadamard_img_tensor256.mat");
% V_hd = ws_hdv.V_hd256;

% ws = load('sobol_pairs_mul_xnor1k.mat','x2_1k');% Loads the matrix of MAE using xor operator for sobol sequences
% x2_1k = ws.x2_1k;
% a1k = find(x2_1k(1,:) ~= 0); %Find worst case sobol sequence indices
% N_sobol = 1:1111;   %Vector of sobol sequence indices
% dd1 = setdiff(N_sobol,a1k);

%**************************Mehran_New**************************************
%yy = ones(total_test_images,image_row_size*image_column_size,D);
%xx = ones(total_training_images,image_row_size*image_column_size,D);

N_sobol = 1:1111;   %Vector of sobol sequence indices

% ws = load('sobol_pairs_mul_xnor1k.mat','x2_1k');% Loads the matrix of MAE using xor operator for sobol sequences
% x2_1k = ws.x2_1k;
% a1k = find(x2_1k(1,:) ~= 0); %Find worst case sobol sequence indices
% dd = setdiff(N_sobol,a1k);
% aa = [1,dd];
% 
% aa = aa(1:image_row_size*image_column_size);

% ws2k = load('sobol_pairs_mul_xnor2k.mat','x2_2k');% Loads the matrix of MAE using xor operator for sobol sequences
% x2_2k = ws2k.x2_2k;
% a2k = find(x2_2k(1,:) ~= 0); %Find worst case sobol sequence indices
% dd2k = setdiff(N_sobol,a2k);
% aa2k = [1,dd2k];

% ws8k = load('sobol_pairs_mul_xnor8k.mat','x2_8k');
% x2_8k = ws8k.x2_8k;
% a8k = find(x2_8k(1,:) ~= 0);
% dd8k = setdiff(N_sobol,a8k);
% aa8k = [1,dd8k];
% aa8k = randperm(numel(aa8k));
% aa8k1 = aa8k;

% ws8k1 = load('sobol_bl_optimized_8k_91.mat','aa8k1');
% aa8k = ws8k1.aa8k1;
% 
ws16k1 = load('sobol_bl_optimized_16k_91.mat','aa8k1');
aa16k = ws16k1.aa8k1;




for k = 2:log2(D)
    vd(:,k-1) = vdcorput((D-1),pow2(k-1));
end

sobol_seq1 = net(sobolset(1111), D);
sobol_seq_new = sobol_seq1(:,aa16k);
sobol_seq_new = sobol_seq_new(:,1:135);
sobol_seq_new = [sobol_seq_new,vd];


% HT = net(haltonset(image_row_size*image_column_size),D);
% HT_new = HT(:,1:135);
% HT_new = [HT_new,vd];
% sobol_seq_new = HT_new(:,randperm(image_row_size*image_column_size));



%sobol_seq_new = sobol_seq_new(:,randperm(image_row_size*image_column_size));

% ws16k = load('sobol_pairs_mul_xnor16k.mat','x2_16k');
% x2_16k = ws16k.x2_16k;
% a16k = find(x2_16k ~= 0);
% dd16k = setdiff(N_sobol,a16k);
% aa16k = [1,dd16k];
% aa16k = randperm(numel(aa16k));
% aa8k1 = aa16k;

%dd8k = dd8k(1:image_row_size*image_column_size);

% sobol_seq1 = net(sobolset(1111), D);
% sobol_seq_new = sobol_seq1(:,aa16k);

% sobol_seq1 = net(sobolset(1111), D);
% sobol_seq_new = sobol_seq1(:,aa16k);


% rng(5755);
% sobol_seq_new = lhsdesign(1024,784);
% alpha = (sqrt(5) - 1) / 2; 
% weyl = mod((1:D)*alpha, 1);
% ww = zeros(D,144);
% for i = 1:144
%     ww(:,i) = weyl(randperm(D));
% end
% sobol_seq_new = ww;

%sobol_seq = transpose(sobol_seq_new);
%sobol_reshape = reshape(sobol_seq, [image_row_size,image_column_size,D]);

% seq = zadoffChuSeq(38,1023);
% re = real(seq);
% re(1024) = -0.01;

% a = zeros(1,1023);
% %f = factor(1023);
% for i =2:1023
%     if mod(i,3) ~= 0 && mod(i,11) ~= 0 && mod(i,31) ~= 0
%         if mod(1023,i) ~= 0
%             a(i) = i;
%         end
%     end
% end
% b = find(a ~= 0);
% 
% % f = factor(1023);
% % b = 1:1023;
% % a = setdiff(b,f);
% % a = a(2:numel(a));
% 
% re = zeros(1023,numel(b));
% for zd = 1:numel(b)
%     re(:,zd) = real(zadoffChuSeq(b(zd),1023));
% end
% re(1024,:) = -0.01;

% a2 = zeros(1,2047);
% %f = factor(1023);
% for i =2:2047
%     if mod(i,23) ~= 0 && mod(i,89) ~= 0
%         if mod(2047,i) ~= 0
%             a2(i) = i;
%         end
%     end
% end
% b2 = find(a2 ~= 0);
% 
% re = zeros(2047,numel(b2));
% for zd = 1:numel(b2)
%     re(:,zd) = real(zadoffChuSeq(b2(zd),2047));
% end
% re(2048,:) = -0.01;

a3 = zeros(1,16383);
for t = 2:16383
    if (mod(t,3) ~= 0) && (mod(t,43) ~= 0) && (mod(t,127) ~= 0)
        if mod(16383,t) ~= 0
            a3(t) = t;
        end
    end
end
b3 = find(a3 ~= 0);

re = zeros(16383,numel(b3));
for zd = 1:numel(b3)
    re(:,zd) = real(zadoffChuSeq(b3(zd),16383));
end
re(16384,:) = -0.01;
re = re(8000:16191,:); %for MNIST 1k 7000-8023  2000-11999
%for BloodMNIST 3000-4023
% % for 8k : 8000-16191

%re = sobol_seq1(:,aa16k);

% ws_re = load('re_16k.mat');
% re = ws_re.re;

% a4 = zeros(1,32767);
% for t = 2:32767
%     if (mod(t,7) ~= 0) && (mod(t,31) ~= 0) && (mod(t,151) ~= 0)
%         if mod(32767,t) ~= 0
%             a4(t) = t;
%         end
%     end
% end
% b4 = find(a4 ~= 0);
% 
% re = zeros(32767,numel(b4));
% for zd = 1:numel(b4)
%     re(:,zd) = real(zadoffChuSeq(b4(zd),32767));
% end
% re(32768,:) = -0.01;
% re = re(7000:8023,:); %7000-8023


% re = zeros(8191,image_row_size*image_column_size);
% for zd = 1:(image_row_size*image_column_size)
%     re(:,zd) = real(zadoffChuSeq(zd,8191));
% end
% re(8192,:) = -0.01;


%delta = 0.25;
%criteria = 0.23;

%criteria_train = zeros(image_row_size*image_column_size, total_training_images);
%criteria_test = zeros(image_row_size*image_column_size, total_test_images);

% MNIST HOG [8 8] ---> 12x12 size
ws_hog = load('hog_144.mat');
out_train = ws_hog.out_train;
out_test = ws_hog.out_test;

% BloodMNIST HOG [8 8] ---> 12x12 size
% ws_hog_blood = load('hog144_blood.mat');
% out_train = ws_hog_blood.out_train;
% out_test = ws_hog_blood.out_test;
% out_train = images_train1;
% out_test = images_test1;


% HOG [10 10] ---> 6x6 size
% ws_hog = load('hog_36.mat');
% out_train = ws_hog.out_train;
% out_test = ws_hog.out_test;

% HOG [6 6] ---> 18x18 size
% ws_hog = load('hog_324.mat');
% out_train = ws_hog.out_train;
% out_test = ws_hog.out_test;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       Training with Quantization Start
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% out_train = zeros((image_row_size*image_column_size),total_training_images);
% out_test = zeros((image_row_size*image_column_size),total_test_images);
%Extracting HOG features of MNIST training images

% for index0 = 1:total_training_images
% %     %out_train(:,:,index0) = feature_extractmeh(images_train(:,:,index0));
% %     rr0 = images_train(:,index0);
% %     rr0 = reshape(rr0,[28,28]);
% %     out_extract_train = extractHOGFeatures(rr0, 'CellSize', [8 8]);
% %     out_extract_train1 = transpose(out_extract_train);
% %     %out_extract_train1 = out_extract_train1(randperm(numel(out_extract_train1)));
% %     %out_extract_train1 = reshape(out_extract_train, [12,12]);
% %     %out_extract_train1 = rescale(out_extract_train1, 0, 1);
% %     out_train(:,index0) = out_extract_train1;
% 
%     delta = (max(out_train(:,index0)) - min(out_train(:,index0)))/32;
%     for i = 1:image_row_size*image_column_size
% 
%         if out_train(i,index0)/delta <= (ceil(out_train(i,index0)/delta) + floor(out_train(i,index0)/delta))/2
%             criteria_train(i,index0) = floor(out_train(i,index0)/delta) * delta;
%         else
%             criteria_train(i,index0) = ceil(out_train(i,index0)/delta) * delta;
%         end
% 
%         for z = 1:D
%               %if out_train(i,index0) < 0.23
%               if criteria_train(i,index0) < 0.16
%                   if 0.5 <= sobol_seq_new(z,i)
%                   %if 0.5 <= re(z,i)
%                     %if 0.5 <= sobol_reshape(i,j,z)
%                         xx(index0,i,z) = -1;
% %                   else
% %                       xx(index0,i,z) = 1;
%                   end
%               %elseif out_train(i,index0) <= sobol_seq_new(z,i)
%               %elseif out_train(i,index0) <= re(z,i)
%               elseif criteria_train(i,index0) <= re(z,i)
%                     xx(index0,i,z) = -1;
% %               else
% %                   xx(index0,i,z) = 1;
%               end
%         end
%     end
%     
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       Training with Quantization End
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       Testing with Quantization Start
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Extracting HOG features of MNIST test images

% for index1 = 1:total_test_images
% %     %out_test(:,:,index) = feature_extractmeh(images_test(:,:,index));
% %     rr1 = images_test(:,index1);
% %     rr1 = reshape(rr1,[28,28]);
% %     out_extract_test = extractHOGFeatures(rr1, 'CellSize', [8 8]);
% %     out_extract_test1 = transpose(out_extract_test);
% %     %out_extract_test1 = out_extract_test1(randperm(numel(out_extract_test1)));
% %     %out_extract_test1 = reshape(out_extract_test, [12,12]);
% %     %out_extract_test1 = rescale(out_extract_test1, low_intensity, high_intensity);
% %     %out_extract_test1 = rescale(out_extract_test1, 0, 1);
% %     out_test(:,index1) = out_extract_test1;
% 
%     delta = (max(out_test(:,index1)) - min(out_test(:,index1)))/80;
%     for i = 1:image_row_size*image_column_size
% 
%         if out_test(i,index1)/delta <= (ceil(out_test(i,index1)/delta) + floor(out_test(i,index1)/delta))/2
%             criteria_test(i,index1) = floor(out_test(i,index1)/delta) * delta;
%         else
%             criteria_test(i,index1) = ceil(out_test(i,index1)/delta) * delta;
%         end
% 
%         for z = 1:D
%             %if out_test(i,index1) < 0.23
%             if criteria_test(i,index1) < 0.1
%                 if 0.5 <= sobol_seq_new(z,i)
%                 %if 0.5 <= re(z,i)
%                         yy(index1,i,z) = -1;
% %                 else
% %                     yy(index1,i,z) = 1;
%                 end
%             %elseif out_test(i,index1) <= sobol_seq_new(z,i)
%             %elseif out_test(i,index1) <= re(z,i)
%             elseif criteria_test(i,index1) <= re(z,i)
%                     yy(index1,i,z) = -1;
% %             else
% %                 yy(index1,i,z) = 1;
%             end
%         end
%         
%     end
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       Testing with Quantization End
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       Inference without Quantization Start
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for index1 = 1:total_test_images
%     %out_test(:,:,index) = feature_extractmeh(images_test(:,:,index));
% %     rr1 = images_test(:,index1);
% %     rr1 = reshape(rr1,[28,28]);
% %     out_extract_test = extractHOGFeatures(rr1, 'CellSize', [8 8]);
% %     out_extract_test1 = transpose(out_extract_test);
% %     %out_extract_test1 = out_extract_test1(randperm(numel(out_extract_test1)));
% %     %out_extract_test1 = reshape(out_extract_test, [12,12]);
% %     %out_extract_test1 = rescale(out_extract_test1, low_intensity, high_intensity);
% %     %out_extract_test1 = rescale(out_extract_test1, 0, 1);
% %     out_test(:,index1) = out_extract_test1;
%     for i = 1:image_row_size*image_column_size
%         for z = 1:D
%             if out_test(i,index1) < 0.16
%                 if 0.5 <= sobol_seq_new(z,i)
%                 %if 0.5 <= re(z,i)
%                         yy(index1,i,z) = -1;
%                 end
%             %elseif out_test(i,index1) <= sobol_seq_new(z,i)
%             elseif out_test(i,index1) <= re(z,i)
%                     yy(index1,i,z) = -1;
%             end
%         end
%         
%     end
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       Inference without Quantization End
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% for index0 = 1:total_training_images
%     out_train(:,index0) = rescale(images_train(:,index0),0,1);
%     for i = 1:image_row_size*image_column_size
%         for z = 1:D
%             if out_train(i,index0) == 0
%                 if 0.5 <= sobol_seq_new(z,i)
%                     xx(index0,i,z) = -1;
%                 end
%             elseif out_train(i,index0) <= sobol_seq_new(z,i)
%                 xx(index0,i,z) = -1;
%             end
%         end
%     end
% end
% for index1 = 1:total_test_images
%     out_test(:,index1) = rescale(images_test(:,index1),0,1);
%     for i = 1:image_row_size*image_column_size
%         for z = 1:D
%             if out_test(i,index1) == 0
%                 if 0.5 <= sobol_seq_new(z,i)
%                     yy(index1,i,z) = -1;
%                 end
%             elseif out_test(i,index1) <= sobol_seq_new(z,i)
%                 yy(index1,i,z) = -1;
%             end
%         end
%     end
% end

%**************************Mehran_New**************************************

% bitstream_f = false(1,10);
% bitstream_sob = false(1,10);


%tic
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

    %shaped_images = shapedim12_mex(images_train1(:,TRAIN_IMAGE_INDEX));


    %shaped_images = uint8(shaped_images);

    %shaped_images(1,:,:) = reshape(images_train(:, TRAIN_IMAGE_INDEX), [1,28,28]);

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
        %xored_Images = binding1k_12_128_mex(image_row_size,image_column_size,D,P_hypervector,intensity_vector,shaped_images);
        %xored_Images = output_sobol(images_train1(:,TRAIN_IMAGE_INDEX),TRAIN_IMAGE_INDEX);
        %xored_Images = output_hadamard(shaped_images,V_hd);
        %xored_Images = reshape(xx(TRAIN_IMAGE_INDEX,:,:),image_row_size,image_column_size,D);

        xx = ones(image_row_size*image_column_size, D);
        criteria_train = zeros(1,image_row_size*image_column_size);
        criteria_sobol = zeros(D,image_row_size*image_column_size);
        criteria_zdchu = zeros(D,image_row_size*image_column_size);

        for j = 1:image_row_size*image_column_size
            delta_sobol = (max(sobol_seq_new(:,j)) - min(sobol_seq_new(:,j)))/32; %40 16       192
            delta_zdchu = (max(re(:,j)) - min(abs(re(:,j))))/32; %8 16
            for zz = 1:D
                % Creating sobol criteria
                if sobol_seq_new(zz,j)/delta_sobol <= (ceil(sobol_seq_new(zz,j))/delta_sobol + floor(sobol_seq_new(zz,j))/delta_sobol)/2
                    criteria_sobol(zz,j) = floor((sobol_seq_new(zz,j))/delta_sobol)*delta_sobol;
                else
                    criteria_sobol(zz,j) = ceil((sobol_seq_new(zz,j))/delta_sobol)*delta_sobol;
                end
                %Creating zdchu criteria
                if re(zz,j)/delta_zdchu <= (ceil(re(zz,j))/delta_zdchu + floor(re(zz,j))/delta_zdchu)/2
                    criteria_zdchu(zz,j) = floor((re(zz,j))/delta_zdchu)*delta_zdchu;
                else
                    criteria_zdchu(zz,j) = ceil((re(zz,j))/delta_zdchu)*delta_zdchu;
                end

            end
        end

        delta = (max(out_train(:,TRAIN_IMAGE_INDEX)) - min(out_train(:,TRAIN_IMAGE_INDEX)))/32; %80
        for i = 1:image_row_size*image_column_size
            if out_train(i,TRAIN_IMAGE_INDEX)/delta <= (ceil(out_train(i,TRAIN_IMAGE_INDEX)/delta) + floor(out_train(i,TRAIN_IMAGE_INDEX)/delta))/2
                criteria_train(i) = floor(out_train(i,TRAIN_IMAGE_INDEX)/delta) * delta;
            else
                criteria_train(i) = ceil(out_train(i,TRAIN_IMAGE_INDEX)/delta) * delta;
            end
            for z = 1:D
%                 if criteria_train(i) < 0.1
%                     if 0.5 <= sobol_seq_new(z,i)
%                         xx(i,z) = -1;
%                     end
%                 elseif criteria_train(i) <= re(z,i)
%                     xx(i,z) = -1;
%                 end
%                 if criteria_train(i) <= 0.1
%                     if 0.5 <= criteria_sobol(z,i)
%                         xx(i,z) = -1;
%                     elseif criteria_train(i) <= re(z,i)
%                         xx(i,z) = -1;
%                     end
%                 
%                 end

                if  criteria_sobol(z,i) <= criteria_train(i) %out_train(i,TRAIN_IMAGE_INDEX)
                    xx(i,z) = -1;
                elseif criteria_train(i) <= criteria_zdchu(z,i)
                    xx(i,z) = -1;
                end


            end
        end

%**************************************************************************
% delta = (max(out_train(:,TRAIN_IMAGE_INDEX)) - min(out_train(:,TRAIN_IMAGE_INDEX)))/80; %80
%         for i = 1:image_row_size*image_column_size
%             if out_train(i,TRAIN_IMAGE_INDEX)/delta <= (ceil(out_train(i,TRAIN_IMAGE_INDEX)/delta) + floor(out_train(i,TRAIN_IMAGE_INDEX)/delta))/2
%                 criteria_train(i) = floor(out_train(i,TRAIN_IMAGE_INDEX)/delta) * delta;
%             else
%                 criteria_train(i) = ceil(out_train(i,TRAIN_IMAGE_INDEX)/delta) * delta;
%             end
%         end

% tt = rescale(out_train(:,TRAIN_IMAGE_INDEX),0,1);
% 
% for i =1:image_row_size*image_column_size
%     for z = 1:D
%         if tt(i) <= mean(tt)
%             if mean(sobol_seq_new(i)) <= sobol_seq_new(z,i)
%                 xx(i,z) = -1;
%             end
%         elseif tt(i) <= re(z,i)
%             xx(i,z) = -1;
%         end
%     end
% end



%**************************************************************************
        
% for i = 1:image_row_size*image_column_size
%     if out_train(i,TRAIN_IMAGE_INDEX) <= 0.1
%         bitstream_f(i,:) = logical([0 0 0 0 0 0 0 0 0 1]);
%     elseif 0.1 < out_train(i,TRAIN_IMAGE_INDEX) && out_train(i,TRAIN_IMAGE_INDEX) <= 0.2
%         bitstream_f(i,:) = logical([0 0 0 0 0 0 0 0 1 1]);
%     elseif 0.2 < out_train(i,TRAIN_IMAGE_INDEX) && out_train(i,TRAIN_IMAGE_INDEX) <= 0.3
%         bitstream_f(i,:) = logical([0 0 0 0 0 0 0 1 1 1]);
%     elseif 0.3 < out_train(i,TRAIN_IMAGE_INDEX) && out_train(i,TRAIN_IMAGE_INDEX) <= 0.4
%         bitstream_f(i,:) = logical([0 0 0 0 0 0 1 1 1 1]);
%     elseif 0.4 < out_train(i,TRAIN_IMAGE_INDEX) && out_train(i,TRAIN_IMAGE_INDEX) <= 0.5
%         bitstream_f(i,:) = logical([0 0 0 0 0 1 1 1 1 1]);
%     elseif 0.5 < out_train(i,TRAIN_IMAGE_INDEX) && out_train(i,TRAIN_IMAGE_INDEX) <= 0.6
%         bitstream_f(i,:) = logical([0 0 0 0 1 1 1 1 1 1]);
%     elseif 0.6 < out_train(i,TRAIN_IMAGE_INDEX) && out_train(i,TRAIN_IMAGE_INDEX) <= 0.7
%         bitstream_f(i,:) = logical([0 0 0 1 1 1 1 1 1 1]);
%     elseif 0.7 < out_train(i,TRAIN_IMAGE_INDEX) && out_train(i,TRAIN_IMAGE_INDEX) <= 0.8
%         bitstream_f(i,:) = logical([0 0 1 1 1 1 1 1 1 1]);
%     elseif 0.8 < out_train(i,TRAIN_IMAGE_INDEX) && out_train(i,TRAIN_IMAGE_INDEX) <= 0.9
%         bitstream_f(i,:) = logical([0 1 1 1 1 1 1 1 1 1]);
%     else
%         bitstream_f(i,:) = logical([1 1 1 1 1 1 1 1 1 1]);
%     end
%     for z = 1:D
%         if sobol_seq_new(z,i) <= 0.1
%             bitstream_sob(z,:) = logical([0 0 0 0 0 0 0 0 0 1]);
%         elseif 0.1 < sobol_seq_new(z,i) && sobol_seq_new(z,i) <= 0.2
%             bitstream_sob(z,:) = logical([0 0 0 0 0 0 0 0 1 1]);
%         elseif 0.2 < sobol_seq_new(z,i) && sobol_seq_new(z,i) <= 0.3
%             bitstream_sob(z,:) = logical([0 0 0 0 0 0 0 1 1 1]);
%         elseif 0.3 < sobol_seq_new(z,i) && sobol_seq_new(z,i) <= 0.4
%             bitstream_sob(z,:) = logical([0 0 0 0 0 0 1 1 1 1]);
%         elseif 0.4 < sobol_seq_new(z,i) && sobol_seq_new(z,i) <= 0.5
%             bitstream_sob(z,:) = logical([0 0 0 0 0 1 1 1 1 1]);
%         elseif 0.5 < sobol_seq_new(z,i) && sobol_seq_new(z,i) <= 0.6
%             bitstream_sob(z,:) = logical([0 0 0 0 1 1 1 1 1 1]);
%         elseif 0.6 < sobol_seq_new(z,i) && sobol_seq_new(z,i) <= 0.7
%             bitstream_sob(z,:) = logical([0 0 0 1 1 1 1 1 1 1]);
%         elseif 0.7 < sobol_seq_new(z,i) && sobol_seq_new(z,i) <= 0.8
%             bitstream_sob(z,:) = logical([0 0 1 1 1 1 1 1 1 1]);
%         elseif 0.8 < sobol_seq_new(z,i) && sobol_seq_new(z,i) <= 0.9
%             bitstream_sob(z,:) = logical([0 1 1 1 1 1 1 1 1 1]);
%         else
%             bitstream_sob(z,:) = logical([1 1 1 1 1 1 1 1 1 1]);
%         end
%         if or(not(bitstream_sob(z,:)),and(bitstream_f(i,:),bitstream_sob(z,:))) == logical([1 1 1 1 1 1 1 1 1 1])
%             xx(i,z) = -1;
%         else
%             xx(i,z) = 1;
%         end
%     end
% 
% end
        
        
        xored_Images = reshape(xx,image_row_size,image_column_size,D);

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
    bundled = zeros(1, D);
    reshaped_xored_Images = zeros(1, D);

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
    
    %bundled = bundling1k_mex(image_row_size,image_column_size,D,xored_Images);
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
%toc

%WaitMessage.Destroy %close status bar

%OR--------------------------------------------------------------------
cumulative_class_hypervector = cat(1, cumulative_class_hypervector0, cumulative_class_hypervector1, cumulative_class_hypervector2, ...
    cumulative_class_hypervector3, cumulative_class_hypervector4, cumulative_class_hypervector5, cumulative_class_hypervector6, ...
    cumulative_class_hypervector7, cumulative_class_hypervector8, cumulative_class_hypervector9);

% cumulative_class_hypervector = cat(1, cumulative_class_hypervector0, cumulative_class_hypervector1, cumulative_class_hypervector2, ...
%     cumulative_class_hypervector3, cumulative_class_hypervector4, cumulative_class_hypervector5, cumulative_class_hypervector6, ...
%     cumulative_class_hypervector7);
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

%tic
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
    %shaped_images = shapedim12_mex(images_test1(:,TESTING_IMAGE_INDEX));
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
    %xored_Images = binding1k_12_128_mex(image_row_size,image_column_size,D,P_hypervector,intensity_vector,shaped_images);
    %xored_Images = output_sobol(images_test1(:,TESTING_IMAGE_INDEX),TESTING_IMAGE_INDEX);
    %xored_Images = output_hadamard(shaped_images,V_hd);
    %xored_Images = reshape(yy(TESTING_IMAGE_INDEX,:,:),image_row_size,image_column_size,D);

    yy = ones(image_row_size*image_column_size,D);

    criteria_test = zeros(1,image_row_size*image_column_size);
    %delta = (max(out_test(:,index1)) - min(out_test(:,index1)))/80;

        criteria_sobol = zeros(D,image_row_size*image_column_size);
        criteria_zdchu = zeros(D,image_row_size*image_column_size);

        for j = 1:image_row_size*image_column_size
            delta_sobol = (max(sobol_seq_new(:,j)) - min(sobol_seq_new(:,j)))/32; %40
            delta_zdchu = (max(re(:,j)) - min(abs(re(:,j))))/32; %8
            for zz = 1:D
                if sobol_seq_new(zz,j)/delta_sobol <= (ceil(sobol_seq_new(zz,j))/delta_sobol + floor(sobol_seq_new(zz,j))/delta_sobol)/2
                    criteria_sobol(zz,j) = floor((sobol_seq_new(zz,j))/delta_sobol)*delta_sobol;
                else
                    criteria_sobol(zz,j) = ceil((sobol_seq_new(zz,j))/delta_sobol)*delta_sobol;
                end

                 %Creating zdchu criteria
                if re(zz,j)/delta_zdchu <= (ceil(re(zz,j))/delta_zdchu + floor(re(zz,j))/delta_zdchu)/2
                    criteria_zdchu(zz,j) = floor((re(zz,j))/delta_zdchu)*delta_zdchu;
                else
                    criteria_zdchu(zz,j) = ceil((re(zz,j))/delta_zdchu)*delta_zdchu;
                end               
            end
        end

%     for i = 1:image_row_size*image_column_size
%         for z = 1:D
%             if out_test(i,TESTING_IMAGE_INDEX) < 0.1
%                 if 0.5 <= sobol_seq_new(z,i)
%                     yy(i,z) = -1;
%                 end
%             elseif out_test(i,TESTING_IMAGE_INDEX) <= re(z,i)
%                 yy(i,z) = -1;
%             end
% %               if criteria_sobol(z,i) <= out_test(i,TESTING_IMAGE_INDEX)
% %                   yy(i,z) = -1;
% %               elseif out_test(i,TESTING_IMAGE_INDEX) <= criteria_zdchu(z,i)
% %                   yy(i,z) = -1;
% %               end
%         end
%     end




    delta = (max(out_test(:,TESTING_IMAGE_INDEX)) - min(out_test(:,TESTING_IMAGE_INDEX)))/32;
    for i = 1:image_row_size*image_column_size

        if out_test(i,TESTING_IMAGE_INDEX)/delta <= (ceil(out_test(i,TESTING_IMAGE_INDEX)/delta) + floor(out_test(i,TESTING_IMAGE_INDEX)/delta))/2
            criteria_test(i) = floor(out_test(i,TESTING_IMAGE_INDEX)/delta) * delta;
        else
            criteria_test(i) = ceil(out_test(i,TESTING_IMAGE_INDEX)/delta) * delta;
        end

        for z = 1:D
            %if out_test(i,index1) < 0.23
            if criteria_sobol(z,i) <= criteria_test(i)
                  yy(i,z) = -1;
            elseif criteria_test(i) <= criteria_zdchu(z,i)
                  yy(i,z) = -1;
            end
        end
        
    end

%**************************************************************************
% uu = rescale(out_test(:,TESTING_IMAGE_INDEX),0,1);
% 
% for i =1:image_row_size*image_column_size
%     for z = 1:D
%         if uu(i) < sobol_seq_new(z,i)
%             yy(i,z) = -1;
%         end
%     end
% end



%**************************************************************************



% for i = 1:image_row_size*image_column_size
%     if out_test(i,TESTING_IMAGE_INDEX) <= 0.1
%         bitstream_f(i,:) = logical([0 0 0 0 0 0 0 0 0 1]);
%     elseif 0.1 < out_test(i,TESTING_IMAGE_INDEX) && out_test(i,TESTING_IMAGE_INDEX) <= 0.2
%         bitstream_f(i,:) = logical([0 0 0 0 0 0 0 0 1 1]);
%     elseif 0.2 < out_test(i,TESTING_IMAGE_INDEX) && out_test(i,TESTING_IMAGE_INDEX) <= 0.3
%         bitstream_f(i,:) = logical([0 0 0 0 0 0 0 1 1 1]);
%     elseif 0.3 < out_test(i,TESTING_IMAGE_INDEX) && out_test(i,TESTING_IMAGE_INDEX) <= 0.4
%         bitstream_f(i,:) = logical([0 0 0 0 0 0 1 1 1 1]);
%     elseif 0.4 < out_test(i,TESTING_IMAGE_INDEX) && out_test(i,TESTING_IMAGE_INDEX) <= 0.5
%         bitstream_f(i,:) = logical([0 0 0 0 0 1 1 1 1 1]);
%     elseif 0.5 < out_test(i,TESTING_IMAGE_INDEX) && out_test(i,TESTING_IMAGE_INDEX) <= 0.6
%         bitstream_f(i,:) = logical([0 0 0 0 1 1 1 1 1 1]);
%     elseif 0.6 < out_test(i,TESTING_IMAGE_INDEX) && out_test(i,TESTING_IMAGE_INDEX) <= 0.7
%         bitstream_f(i,:) = logical([0 0 0 1 1 1 1 1 1 1]);
%     elseif 0.7 < out_test(i,TESTING_IMAGE_INDEX) && out_test(i,TESTING_IMAGE_INDEX) <= 0.8
%         bitstream_f(i,:) = logical([0 0 1 1 1 1 1 1 1 1]);
%     elseif 0.8 < out_test(i,TESTING_IMAGE_INDEX) && out_test(i,TESTING_IMAGE_INDEX) <= 0.9
%         bitstream_f(i,:) = logical([0 1 1 1 1 1 1 1 1 1]);
%     else
%         bitstream_f(i,:) = logical([1 1 1 1 1 1 1 1 1 1]);
%     end
%     for z = 1:D
%         if sobol_seq_new(z,i) <= 0.1
%             bitstream_sob(z,:) = logical([0 0 0 0 0 0 0 0 0 1]);
%         elseif 0.1 < sobol_seq_new(z,i) && sobol_seq_new(z,i) <= 0.2
%             bitstream_sob(z,:) = logical([0 0 0 0 0 0 0 0 1 1]);
%         elseif 0.2 < sobol_seq_new(z,i) && sobol_seq_new(z,i) <= 0.3
%             bitstream_sob(z,:) = logical([0 0 0 0 0 0 0 1 1 1]);
%         elseif 0.3 < sobol_seq_new(z,i) && sobol_seq_new(z,i) <= 0.4
%             bitstream_sob(z,:) = logical([0 0 0 0 0 0 1 1 1 1]);
%         elseif 0.4 < sobol_seq_new(z,i) && sobol_seq_new(z,i) <= 0.5
%             bitstream_sob(z,:) = logical([0 0 0 0 0 1 1 1 1 1]);
%         elseif 0.5 < sobol_seq_new(z,i) && sobol_seq_new(z,i) <= 0.6
%             bitstream_sob(z,:) = logical([0 0 0 0 1 1 1 1 1 1]);
%         elseif 0.6 < sobol_seq_new(z,i) && sobol_seq_new(z,i) <= 0.7
%             bitstream_sob(z,:) = logical([0 0 0 1 1 1 1 1 1 1]);
%         elseif 0.7 < sobol_seq_new(z,i) && sobol_seq_new(z,i) <= 0.8
%             bitstream_sob(z,:) = logical([0 0 1 1 1 1 1 1 1 1]);
%         elseif 0.8 < sobol_seq_new(z,i) && sobol_seq_new(z,i) <= 0.9
%             bitstream_sob(z,:) = logical([0 1 1 1 1 1 1 1 1 1]);
%         else
%             bitstream_sob(z,:) = logical([1 1 1 1 1 1 1 1 1 1]);
%         end
%         if or(not(bitstream_sob(z,:)),and(bitstream_f(i,:),bitstream_sob(z,:))) == logical([1 1 1 1 1 1 1 1 1 1])
%             yy(i,z) = -1;
%         else
%             yy(i,z) = 1;
%         end
%     end
% 
% end


%     delta = (max(out_test(:,TESTING_IMAGE_INDEX)) - min(out_test(:,TESTING_IMAGE_INDEX)))/16;
%     for i = 1:image_row_size*image_column_size
% 
%         if out_test(i,TESTING_IMAGE_INDEX)/delta <= (ceil(out_test(i,TESTING_IMAGE_INDEX)/delta) + floor(out_test(i,TESTING_IMAGE_INDEX)/delta))/2
%             criteria_test(i) = floor(out_test(i,TESTING_IMAGE_INDEX)/delta) * delta;
%         else
%             criteria_test(i) = ceil(out_test(i,TESTING_IMAGE_INDEX)/delta) * delta;
%         end
% 
%         for z = 1:D
%             %if out_test(i,index1) < 0.23
%             if criteria_test(i) < 0.1
%                 if 0.5 <= sobol_seq_new(z,i)
%                 %if 0.5 <= re(z,i)
%                         yy(i,z) = -1;
% %                 else
% %                     yy(index1,i,z) = 1;
%                 end
%             %elseif out_test(i,index1) <= sobol_seq_new(z,i)
%             %elseif out_test(i,index1) <= re(z,i)
%             elseif criteria_test(i) <= re(z,i)
%                     yy(i,z) = -1;
% %             else
% %                 yy(index1,i,z) = 1;
%             end
%         end
%         
%     end


    xored_Images = reshape(yy,image_row_size,image_column_size,D);

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
    
    bundled = zeros(1, D);
    reshaped_xored_Images = zeros(1, D);
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
    
    %bundled = bundling1k_mex(image_row_size,image_column_size,D,xored_Images);
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
    
    for classes = 1:numberOfClasses
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
%toc
cl_percentage = (accuracy * 100) / total_test_images;
