clear
% classification_percentage = zeros(1,10);
% for iter = 1:10

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


%MNIST
% total_training_images = 60000;
% total_test_images = 10000;
% 
% image_row_size = 12;
% image_column_size = 12;

%CIFAR10
load('cifar10.mat');

total_training_images = 50000;
total_test_images = 10000;

image_row_size = 18;
image_column_size = 18;

% images_train1 = images_train;
% images_test1 = images_test;


load('HOG_CIFAR10.mat');
images_train1 = double(uint8(rescale(featureVectorTRAIN,0,255)));
images_test1 = double(uint8(rescale(featureVectorTEST,0,255)));



% img_r_rescale = 14;
% img_c_rescale = 14;

%Reshaping the training images
% out_shape_train = zeros(28, 28);
% out_shape2 = zeros(img_r_rescale, img_c_rescale, total_training_images);
% for kk = 1:total_training_images
%     for ii = 0:27
%         out_shape_train(:,ii+1) = images_train((ii*28)+1:(ii*28)+28, kk)';
%         %k = k + 28;
%     end
%     out_shape2(:,:,kk) = imresize(out_shape_train, 0.5);
% end
% out_shape2 = reshape(out_shape2, [(img_r_rescale*img_c_rescale),total_training_images]);
% 
% %Reshaping the test images
% out_shape_test = zeros(28, 28);
% out_shape3 = zeros(img_r_rescale, img_c_rescale, total_test_images);
% for kk = 1:total_test_images
%     for ii = 0:27
%         out_shape_test(:,ii+1) = images_train((ii*28)+1:(ii*28)+28, kk)';
%         %k = k + 28;
%     end
%     out_shape3(:,:,kk) = imresize(out_shape_test, 0.5);
% end
% out_shape3 = reshape(out_shape3, [(img_r_rescale*img_c_rescale),total_test_images]);

% images_train1 = uint8(out_shape2);
% images_train1 = double(images_train1);
% images_test1 = uint8(out_shape3);
% images_test1 = double(images_test1);

% MNIST HOG [8 8] ---> 12x12 size
% ws_hog = load('hog_144.mat');
% out_train = ws_hog.out_train;
% out_test = ws_hog.out_test;


% images_train1 = uint8(rescale(out_train,0,255));
% images_train1 = double(images_train1);
% images_test1 = uint8(rescale(out_test,0,255));
% images_test1 = double(images_test1);

%TRAIN_IMAGE_INDEX = 1; %To be parametrized

%trainDB_size = 1; %a dummy parameter
%numberOfClasses = 10;


D = 8192; %vector dimension

%8-bit gray-scale
low_intensity = 0;
high_intensity = 255;

M = high_intensity+1; %quantization interval

initial_vector_seed = ones(1,D);
intensity_vector = ones(M,D);
P_hypervector = ones(image_row_size,image_column_size,D);
%reshaped_P_hv = zeros(1, D);
%coder.varsize("reshaped_P_hv", [1 D]);

%Static threshold for position hypervector vectors, P, orthogonal
threshold = ((high_intensity+1)/2); %Half value of max. intensity value; mid value

%Dynamic threshold parameter for level hypervector vectors, L, correlated
bitflip_count = D/(M); %note that D >= 2*high_intensity
% r = round((high_intensity-low_intensity).*rand(14,14,D) + low_intensity);
% xx = threshold > r;
% P_hypervector = double(xx);
% P_hypervector(xx == 0) = -1;





% for k = 2:log2(D)
%     vd(:,k-1) = vdcorput((D-1),pow2(k-1));
% end




% ss = load('sobol_pairs_mul.mat');
% d = ss.d;
% dd = find(d(5,:) == 0);

ws = load('sobol_pairs_mul_xor.mat','x1');% Loads the matrix of MAE using xor operator for sobol sequences
x1 = ws.x1;
a = find(x1(1,:) ~= 0); %Find worst case sobol sequence indices
N_sobol = 1:1111;   %Vector of sobol sequence indices
dd = setdiff(N_sobol,a);    %Excludes worst case sobol sequence indices from the main sobol sequence indices
%dd = circshift(dd,1);
%dd(1) = 1;
dd = dd(1:image_row_size*image_column_size);

ws2k = load('sobol_pairs_mul_xnor2k.mat','x2_2k');
x2_2k = ws2k.x2_2k;
a2k = find(x2_2k(1,:) ~= 0);
dd2k = setdiff(N_sobol,a2k);
dd2k = dd2k(1:image_row_size*image_column_size);

ws8k = load('sobol_pairs_mul_xnor8k.mat','x2_8k');
x2_8k = ws8k.x2_8k;
a8k = find(x2_8k(1,:) ~= 0);
dd8k = setdiff(N_sobol,a8k);
dd8k = dd8k(1:image_row_size*image_column_size);

ws16k1 = load('sobol_bl_optimized_16k_91.mat','aa8k1');
aa16k = ws16k1.aa8k1;
%aa16k = aa16k(1:image_row_size*image_column_size);


sobol_seq1 = net(sobolset(1111), D);
sobol_seq1 = sobol_seq1(:,aa16k);
%sobol_seq = net(sobolset(28*28), D);
sobol_seq = transpose(sobol_seq1);
%sobol_seq = transpose(sobol_seq1(:,2:785));
%sobol_reshape = reshape(sobol_seq(1:28*28,:), [28,28,D]);
%sobol_reshape = reshape(sobol_seq_new, [28,28,D]);

%sobol_reshape = reshape(sobol_seq, [image_row_size,image_column_size,D]);

% sobol_seq_new = sobol_seq1(:,1:135);
% sobol_seq_new = [sobol_seq_new,vd];
% sobol_seq_new = sobol_seq_new(:,randperm(image_row_size*image_column_size));
%load('sobol_vdc.mat');
for k = 1:log2(D)
    vd(:,k) = vdcorput((D-1),pow2(k));
end

sobol_seq1 = net(sobolset(1111), D);
sobol_seq_new = sobol_seq1(:,aa16k);
%sobol_seq_new = sobol_seq_new(:,1:315);
%sobol_seq_new = [sobol_seq_new,vd];
sobol_seq_new = sobol_seq_new(:,700:1023);



%sobol_seq_new = sobol_seq1(:,1:1024);

% PP_sobol = ones(D,image_row_size*image_column_size);
% for i = 1:image_row_size*image_column_size   
%         for z = 1:D
%             if 0.5 <= sobol_seq_new(z,i)
%                 PP_sobol(z,i) = -1;
%             end            
%         end    
% end
% PP_sobol = PP_sobol';

PP_halton = ones(D,image_row_size*image_column_size);
HT = net(haltonset(image_row_size*image_column_size),D);

for i = 1:image_row_size*image_column_size   
        for z = 1:D
            if 0.5 <= HT(z,i)
                PP_halton(z,i) = -1;
            end            
        end    
end
PP_halton = PP_halton';


%P_hypervector = reshape(PP_sobol,image_row_size,image_column_size,D);
P_hypervector = reshape(PP_halton,image_row_size,image_column_size,D);

%P_hypervector = reshape(G,image_row_size,image_column_size,D);

alpha = pi;
%alpha = exp(1);
weyl = mod((1:D)*alpha, 1);
% ww = zeros(D,image_row_size*image_column_size);
% for i = 1:image_row_size*image_column_size
%     ww(:,i) = weyl(randperm(D));
% end


% for i = 1:28
%     for j = 1:28
%         reshaped_P_hv(1,:) = P_hypervector(i,j,:);
%     end
% end
%}
%intensity_vector = bitflip1k_mex(D,M,initial_vector_seed,intensity_vector);
%intensity_vector = bitflip(D,M,bitflip_count,initial_vector_seed,intensity_vector); % --->
%Accuracy = 85.83 Latin Hypercube
% 786 - 803 - 

for g= 0:high_intensity
    for d = 1:D
        %if (D/M)*(g/D) <= sobol_seq1(d,1025)   % Accuracy = 78.97 Kasami, 84.61 Latin Hypercube
        %if (D/M)*(g/D) <= sobol_seq_new(d,1)
        %if (D/M)*(g/D) <= vd(d,3)
        %if (D/M)*(g/D) <= R2(d,1)
        if (D/M)*(g/D) <= weyl(d)
        %if (D/M)*(g/D) <= nd5(d,1)
        
        %if (D/M)*(g/D) <= Y1(d,image_row_size*image_column_size+1)
         %if (D/M)*(g/D) <= K_Kasami1(d,image_row_size*image_column_size+1)
            intensity_vector(g+1,d) = -1; % ---> Accuracy = 85.8 Latin Hypercube
        end
    end
end
% h2 = h(2:1025,146:401);
% intensity_vector = h2';


%coder.varsize("intensity_vector", [256 D]);

classification_percentage = HDC_CIFAR10(image_row_size, image_column_size, D, images_train1, images_test1,...
    P_hypervector, intensity_vector, labels_train, labels_test, total_training_images, total_test_images)
%end
