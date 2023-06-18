clear
% classification_percentage = zeros(1,50);
% for iter = 1:50

%************************BloodMNIST Start****************************************
% load('bloodmnist.mat')
% 
% [size_train, image_row_size, image_column_size, ~] = size(train_images);
% images_train = zeros(image_row_size*image_column_size, size_train);
% parfor i = 1:size_train
%     img_temp = cat(3, reshape(train_images(i, :, :, 1), [1,image_row_size*image_column_size]), reshape(train_images(i, :, :, 2), [1,image_row_size*image_column_size]), reshape(train_images(i, :, :, 3), [1,image_row_size*image_column_size]));
%     images_train(:,i) = transpose(double(rgb2gray(img_temp)));
% end
% [size_test, image_row_size, image_column_size, ~] = size(test_images);
% images_test = zeros(image_row_size*image_column_size, size_test);
% parfor i = 1:size_test
%     img_temp = cat(3, reshape(test_images(i, :, :, 1), [1,image_row_size*image_column_size]), reshape(test_images(i, :, :, 2), [1,image_row_size*image_column_size]), reshape(test_images(i, :, :, 3), [1,image_row_size*image_column_size]));
%     images_test(:,i) = transpose(double(rgb2gray(img_temp)));
% end
% labels_train = double(train_labels);
% labels_test = double(test_labels);
% numberOfClasses = max(labels_test) + 1; %if 0 is a label; please check that





%************************BloodMNIST End****************************************

%[images_train, images_test, labels_test, labels_train, images_train_SC, images_test_SC]= mnist_db_construct();

%images_train1 = double(images_train(:,1:1000));
%images_test1 = double(images_test(:,1:100));
%images_train1 = double(images_train);
%images_test1 = double(images_test);
%images_train1 = double(images_train(1:784,1:1000));
%images_train1 = double(images_train(:,1:5000));

%coder.varsize('images_train1', [784 1000]);

%images_train1 = out_train;
%images_test1 = out_test;

%images_test1 = double(images_test(1:784,1:100));
%images_test1 = double(images_test(:,1:1000));

%coder.varsize('images_test1', [784 100]);

% images_train1 = double(images_train);
% images_test1 = double(images_test);
load('cifar10.mat');

total_training_images = 50000; %MNIST 60000  11959
total_test_images = 10000; %MNIST 10000  3421

image_row_size = 18;
image_column_size = 18;

D = 1024; %vector dimension

%8-bit gray-scale
low_intensity = 0;
high_intensity = 127;

% N_sobol = 1:1111;
% ws8k = load('sobol_pairs_mul_xnor8k.mat','x2_8k');
% x2_8k = ws8k.x2_8k;
% a8k = find(x2_8k(1,:) ~= 0);
% dd8k = setdiff(N_sobol,a8k);
% aa8k1 = [1,dd8k];
% aa8k1 = randperm(numel(aa8k1));

% yy = ones(total_test_images,image_row_size*image_column_size,D);
% xx = ones(total_training_images,image_row_size*image_column_size,D);
% 
% ws = load('sobol_pairs_mul_xnor1k.mat','x2_1k');% Loads the matrix of MAE using xor operator for sobol sequences
% x2_1k = ws.x2_1k;
% a1k = find(x2_1k(1,:) ~= 0); %Find worst case sobol sequence indices
% N_sobol = 1:1111;   %Vector of sobol sequence indices
% dd = setdiff(N_sobol,a1k);
% aa = [1,dd];
% aa = aa(1:144);
% 
% sobol_seq1 = net(sobolset(1111), D);
% sobol_seq_new = sobol_seq1(:,aa);
% sobol_seq = transpose(sobol_seq_new);
% 
% out_train = zeros((image_row_size*image_column_size),total_training_images);
% out_test = zeros((image_row_size*image_column_size),total_test_images);
% %Extracting HOG features of MNIST training images
% for index0 = 1:total_training_images
%     %out_train(:,:,index0) = feature_extractmeh(images_train(:,:,index0));
%     rr0 = images_train(:,index0);
%     rr0 = reshape(rr0,[28,28]);
%     out_extract_train = extractHOGFeatures(rr0);
%     out_extract_train1 = transpose(out_extract_train);
%     %out_extract_train1 = reshape(out_extract_train, [12,12]);
%     out_extract_train1 = rescale(out_extract_train1, 0, 1);
%     out_train(:,index0) = out_extract_train1;
% 
%     for i = 1:image_row_size*image_column_size
%         for z = 1:D
%             if out_train(i,index0) <= sobol_seq_new(z,i)
%                 xx(index0,i,z) = -1;
%             end
%         end
%     end
% end
% %Extracting HOG features of MNIST test images
% for index1 = 1:total_test_images
%     %out_test(:,:,index) = feature_extractmeh(images_test(:,:,index));
%     rr1 = images_test(:,index1);
%     rr1 = reshape(rr1,[28,28]);
%     out_extract_test = extractHOGFeatures(rr1);
%     out_extract_test1 = transpose(out_extract_test);
%     %out_extract_test1 = reshape(out_extract_test, [12,12]);
%     %out_extract_test1 = rescale(out_extract_test1, low_intensity, high_intensity);
%     out_extract_test1 = rescale(out_extract_test1, 0, 1);
%     out_test(:,index1) = out_extract_test1;
%     for i = 1:image_row_size*image_column_size
%         for z = 1:D
%             if out_test(i,index1) <= sobol_seq_new(z,i)
%                 yy(index1,i,z) = -1;
%             end
%         end
%     end
% end
% 
% 
% 
% images_train1 = uint8(out_train);
% images_train1 = double(images_train1);
% %images_train1 = rescale(images_train1,0,255);
% images_test1 = uint8(out_test);
% images_test1 = double(images_test1);

%TRAIN_IMAGE_INDEX = 1; %To be parametrized

%trainDB_size = 1; %a dummy parameter
%numberOfClasses = 10;


M = high_intensity+1; %quantization interval

initial_vector_seed = ones(1,D);
intensity_vector = ones(M,D);
P_hypervector = ones(image_row_size,image_column_size,D);
reshaped_P_hv = zeros(1, D);
%coder.varsize("reshaped_P_hv", [1 D]);

%Static threshold for position hypervector vectors, P, orthogonal
threshold = ((high_intensity+1)/2); %Half value of max. intensity value; mid value

%Dynamic threshold parameter for level hypervector vectors, L, correlated
bitflip_count = D/(M); %note that D >= 2*high_intensity
% r = round((high_intensity-low_intensity).*rand(image_row_size,image_column_size,D) + low_intensity);
% xx = threshold > r;
% P_hypervector = double(xx);
% P_hypervector(xx == 0) = -1;

%Hadamard Seq
% hadamardseq = hadamard(D);
% for i = 1:image_row_size
%     for j = 1:image_column_size
%         for k = 1:D
%             P_hypervector(i,j,:) = hadamardseq(:,k);
%         end
%     end
% end

% ss = load('sobol_pairs_mul.mat');
% d = ss.d;
% dd = find(d(5,:) == 0);


% ws = load('sobol_pairs_mul_xnor1k.mat','x2_1k');% Loads the matrix of MAE using xor operator for sobol sequences
% x2_1k = ws.x2_1k;
% a1k = find(x2_1k(1,:) ~= 0); %Find worst case sobol sequence indices
% N_sobol = 1:1111;   %Vector of sobol sequence indices
% dd = setdiff(N_sobol,a1k);    %Excludes worst case sobol sequence indices from the main sobol sequence indices
% ww = dd(1:784);
% %dd = circshift(dd,1);
% %dd(1) = 1;
% dd = dd(1:image_row_size*image_column_size);
% 
% ws2k = load('sobol_pairs_mul_xnor2k.mat','x2_2k');
% x2_2k = ws2k.x2_2k;
% a2k = find(x2_2k(1,:) ~= 0);
% dd2k = setdiff(N_sobol,a2k);
% dd2k = dd2k(1:image_row_size*image_column_size);
% 
% ws8k = load('sobol_pairs_mul_xnor8k.mat','x2_8k');
% x2_8k = ws8k.x2_8k;
% a8k = find(x2_8k(1,:) ~= 0);
% dd8k = setdiff(N_sobol,a8k);
% dd8k = dd8k(1:image_row_size*image_column_size);
% 
% sobol_seq1 = net(sobolset(1111), D);
% sobol_seq_new = sobol_seq1(:,dd);
% %sobol_seq = net(sobolset(28*28), D);
% sobol_seq = transpose(sobol_seq_new);
% %sobol_seq = transpose(sobol_seq1(:,2:785));
% %sobol_reshape = reshape(sobol_seq(1:28*28,:), [28,28,D]);
% %sobol_reshape = reshape(sobol_seq_new, [28,28,D]);
% sobol_reshape = reshape(sobol_seq, [image_row_size,image_column_size,D]);
% for i = 0:image_row_size-1
%     for j = 0:image_column_size-1
%         for z = 1:D
%             %if (D/(image_row_size*image_column_size))*((i*28+j)/D) <= sobol_seq(z,8)
%             %if (D/(image_row_size*image_column_size))*((i*28+j)/D) <= sobol_reshape(i+1,j+1,z)
%             if 0.5 <= sobol_reshape(i+1,j+1,z)
%             %if 0.5 <= sobol_seq(z,5)
%                 P_hypervector(i+1,j+1,z) = -1;
%             end
%             
%         end
%     end
% end
% 
% %coder.varsize("P_hypervector", [28 28 D]);
% %{
% for i = 1:28
%     for j = 1:28
%         reshaped_P_hv(1,:) = P_hypervector(i,j,:);
%     end
% end
% %}
% %intensity_vector = bitflip8k_mex(D,M,initial_vector_seed,intensity_vector);
% % 786 - 803 - 
% 
% for g= low_intensity:high_intensity
%     for d = 1:D
%         if (D/M)*(g/D) <= sobol_seq1(d,1) %1099 --->88
%             intensity_vector(g+1,d) = -1;
%         end
%     end
% end


%coder.varsize("intensity_vector", [256 D]);
tic
classification_percentage = HDC_CIFAR10_Q(image_row_size, image_column_size, D,...
    P_hypervector, intensity_vector, labels_train, labels_test, total_training_images, total_test_images)
toc
%end

