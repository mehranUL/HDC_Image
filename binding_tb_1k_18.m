%[images_train, images_test, labels_test, labels_train, images_train_SC, images_test_SC]= mnist_db_construct();

image_row_size = 18;
image_column_size = 18;
%images_train1 = images_train(:,1:1000);
%images_train1 = zeros(784,1000);
D = 1024;
M = 256;
%low_intensity = 0;
%high_intensity = 255;

P_hypervector = zeros(image_row_size,image_column_size,D);
intensity_vector = zeros(M,D);
shaped_images = zeros(image_row_size,image_column_size);
%reshaped_P_hv = zeros(1, 1024);
%coder.varsize("P_hypervector", [28 28 1024]);
%coder.varsize("intensity_vector", [256 1024]);


%{
for it_img = 1:1000
    for ii = 0:27
        shaped_images(:,ii+1) = images_train1((ii*28)+1:(ii*28)+28, it_img)';
        %k = k + 28;
    end
end

r = round((high_intensity-low_intensity).*rand(28,28,D) + low_intensity);
xx = threshold > r;
P_hypervector = double(xx);
P_hypervector(xx == 0) = -1;

intensity_vector = bitflip1k_mex(D,M,initial_vector_seed,intensity_vector);
%}
%tic
out_meh = binding1k_18(image_row_size,image_column_size,D,P_hypervector,intensity_vector,shaped_images);
%toc