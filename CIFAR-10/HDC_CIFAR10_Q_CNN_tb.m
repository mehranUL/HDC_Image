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
% load('cifar10_cnn.mat');
% images_test = images_test_;
% images_train = images_train_;
% labels_train = labels_train_;
% labels_test = labels_test_;
load('cnn_nonscaledmat.mat');

total_training_images = 50000; %MNIST 60000  11959
total_test_images = 10000; %MNIST 10000  3421

image_row_size = 18;
image_column_size = 18;

D = 8192; %vector dimension

%8-bit gray-scale
low_intensity = 0;
high_intensity = 127;



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



% ss = load('sobol_pairs_mul.mat');
% d = ss.d;
% dd = find(d(5,:) == 0);





%coder.varsize("intensity_vector", [256 D]);
tic
classification_percentage = HDC_CIFAR10_Q_CNN(image_row_size, image_column_size, D,images_train,images_test,...
    P_hypervector, intensity_vector, labels_train, labels_test, total_training_images, total_test_images)
toc
%end

