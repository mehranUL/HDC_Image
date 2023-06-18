% classification_percentage = zeros(1,10);
% for iter = 1:10

[images_train, images_test, labels_test, labels_train, images_train_SC, images_test_SC]= mnist_db_construct();
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
total_training_images = 60000;
total_test_images = 10000;

image_row_size = 12;
image_column_size = 12;
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
ws_hog = load('hog_144.mat');
out_train = ws_hog.out_train;
out_test = ws_hog.out_test;


images_train1 = uint8(rescale(out_train,0,255));
images_train1 = double(images_train1);
images_test1 = uint8(rescale(out_test,0,255));
images_test1 = double(images_test1);

%TRAIN_IMAGE_INDEX = 1; %To be parametrized

%trainDB_size = 1; %a dummy parameter
%numberOfClasses = 10;


D = 1024; %vector dimension

%8-bit gray-scale
low_intensity = 0;
high_intensity = 255;

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
% r = round((high_intensity-low_intensity).*rand(14,14,D) + low_intensity);
% xx = threshold > r;
% P_hypervector = double(xx);
% P_hypervector(xx == 0) = -1;

G = gold_code();
G = G(:,1:144);
G = G';

%vd = zeros(D,log2(D));
% for k = 2:log2(D)
%     vd(:,k-1) = vdcorput((D-1),k);
% end

for k = 2:log2(D)
    vd(:,k-1) = vdcorput((D-1),pow2(k-1));
end
%P_hypervector = reshape(G,image_row_size,image_column_size,D);

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

load('sobol_vdc.mat');
%sobol_seq_new = sobol_seq1(:,1:1024);

PP_sobol = ones(D,image_row_size*image_column_size);
for i = 1:image_row_size*image_column_size   
        for z = 1:D
            if 0.5 <= sobol_seq_new(z,i)
                PP_sobol(z,i) = -1;
            end            
        end    
end
PP_sobol = PP_sobol';

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





P_hypervector = reshape(PP_sobol,image_row_size,image_column_size,D);
%P_hypervector = reshape(G,image_row_size,image_column_size,D);

Y1 = lhsdesign(D,image_row_size*image_column_size+1);
Y = Y1(:,1:image_row_size*image_column_size);
Y_tr = transpose(Y);
Y_reshape = reshape(Y_tr, [image_row_size,image_column_size,D]);
% for i = 0:image_row_size-1
%     for j = 0:image_column_size-1
%         for z = 1:D
%             %if (D/(image_row_size*image_column_size))*((i*28+j)/D) <= sobol_seq(z,8)
%             %if (D/(image_row_size*image_column_size))*((i*28+j)/D) <= sobol_reshape(i+1,j+1,z)
%             %if 0.5 <= sobol_reshape(i+1,j+1,z)
%             if 0.5 <= Y_reshape(i+1,j+1,z)
%             
%                 P_hypervector(i+1,j+1,z) = -1;
%             end
%             
%         end
%     end
% end

% K_Kasami1 = kasami(log2(D));
% K_Kasami = K_Kasami1(:,1:image_row_size*image_column_size);
% P_hypervector = reshape(K_Kasami,image_row_size,image_column_size,D);

%Weyl Sequence Start
%alpha = (sqrt(5) - 1) / 2;
%alpha = sqrt(2) - 1;
alpha = pi;
%alpha = exp(1);
weyl = mod((1:D)*alpha, 1);
ww = zeros(D,image_row_size*image_column_size);
for i = 1:image_row_size*image_column_size
    ww(:,i) = weyl(randperm(D));
end

R2_ws = load("R2_1K_144.mat");
R2 = R2_ws.z;

PP_weyl = ones(D,image_row_size*image_column_size);

for i = 1:image_row_size*image_column_size   
        for z = 1:D
            if 0.5 <= R2(z,i)
                PP_weyl(z,i) = -1;
            end            
        end    
end
PP_weyl = PP_weyl';

%P_hypervector = reshape(PP_weyl,image_row_size,image_column_size,D);
%Weyl Sequence End

N = log2(D);
gg = logical(randi(2, [1 N]) - 1);
%
rnd1 = zeros(D,image_row_size*image_column_size);

for i = 1:image_row_size*image_column_size  
    rnd1(1:D/2,i) = lfsr_value512(D/2,i);
    rnd1((D/2+1):D,i) = 1 - rnd1(1:D/2,i);
end

TT = ones(D,image_row_size*image_column_size);

for i = 1:image_row_size*image_column_size
    for z = 1:D
        if 0.5 <= rnd1(z,i)
            TT(z,i) = -1;
        end
    end
end
%P_hypervector = reshape(TT,image_row_size,image_column_size,D);


%Niederreiter Seq Start
% [ nd1, ~ ] = niederreiter2_generate ( 20, 1024, 2, 27 );
% nd1 = nd1';
% PP_nd1 = ones(D,20);
% for i = 1:20    
%         for z = 1:D
%             if 8 <= nd1(z,i)
%                 PP_nd1(z,i) = -1;
%             end            
%         end    
% end
% PP_nd1 = PP_nd1';
% 
% [ nd2, ~ ] = niederreiter2_generate ( 20, 1024, 2, 28 );
% nd2 = nd2';
% PP_nd2 = ones(D,20);
% for i = 1:20    
%         for z = 1:D
%             if 4 <= nd2(z,i)
%                 PP_nd2(z,i) = -1;
%             end            
%         end    
% end
% PP_nd2 = PP_nd2';
% 
% 
% [ nd3, ~ ] = niederreiter2_generate ( 20, 1024, 2, 29 );
% nd3 = nd3';
% PP_nd3 = ones(D,20);
% for i = 1:20    
%         for z = 1:D
%             if 2 <= nd3(z,i)
%                 PP_nd3(z,i) = -1;
%             end            
%         end    
% end
% PP_nd3 = PP_nd3';
% 
% [ nd4, ~ ] = niederreiter2_generate ( 20, 1024, 2, 30 );
% nd4 = nd4';
% PP_nd4 = ones(D,20);
% for i = 1:20    
%         for z = 1:D
%             if 1 <= nd4(z,i)
%                 PP_nd4(z,i) = -1;
%             end            
%         end    
% end
% PP_nd4 = PP_nd4';
% 
% [ nd5, ~ ] = niederreiter2_generate ( 20, 1024, 2, 31 );
% nd5 = nd5';
% PP_nd5 = ones(D,20);
% for i = 1:20    
%         for z = 1:D
%             if 0.5 <= nd5(z,i)
%                 PP_nd5(z,i) = -1;
%             end            
%         end    
% end
% PP_nd5 = PP_nd5';
% 
% [ nd6, ~ ] = niederreiter2_generate ( 20, 1024, 2, 32 );
% nd6 = nd6';
% PP_nd6 = ones(D,20);
% for i = 1:20    
%         for z = 1:D
%             if 0.25 <= nd6(z,i)
%                 PP_nd6(z,i) = -1;
%             end            
%         end    
% end
% PP_nd6 = PP_nd6';
% 
% [ nd7, ~ ] = niederreiter2_generate ( 20, 1024, 2, 33 );
% nd7 = nd7';
% PP_nd7 = ones(D,20);
% for i = 1:20    
%         for z = 1:D
%             if 0.125 <= nd7(z,i)
%                 PP_nd7(z,i) = -1;
%             end            
%         end    
% end
% PP_nd7 = PP_nd7';
% 
% [ nd8, ~ ] = niederreiter2_generate ( 20, 1024, 2, 34 );
% nd8 = nd8';
% PP_nd8 = ones(D,20);
% for i = 1:20    
%         for z = 1:D
%             if 0.0625 <= nd8(z,i)
%                 PP_nd8(z,i) = -1;
%             end            
%         end    
% end
% PP_nd8 = PP_nd8';
% 
% [ nd9, ~ ] = niederreiter2_generate ( 20, 1024, 2, 35 );
% nd9 = nd9';
% PP_nd9 = ones(D,20);
% for i = 1:20    
%         for z = 1:D
%             if 0.03125 <= nd9(z,i)
%                 PP_nd9(z,i) = -1;
%             end            
%         end    
% end
% PP_nd9 = PP_nd9';
% 
% 
% PP_nd = [PP_nd1', PP_nd2', PP_nd3', PP_nd4', PP_nd9', PP_nd6', PP_nd7', PP_nd8']; % nd5 changed to nd9
% nd = [nd1, nd2, nd3, nd4, nd9, nd6, nd7, nd8];
% nd = rescale(nd,0,1);
% PP_nd = PP_nd';
% PP_nd1 = PP_nd(1:144,:);
%P_hypervector = reshape(PP_nd1,image_row_size,image_column_size,D);



h = hadamard(4096);
h1 = h(2:1025,2:145);
h1 = h1';

%P_hypervector = reshape(h1,image_row_size,image_column_size,D);


%PP_R2 = ones(D,image_row_size*image_column_size);


%coder.varsize("P_hypervector", [28 28 D]);
%{
for i = 1:28
    for j = 1:28
        reshaped_P_hv(1,:) = P_hypervector(i,j,:);
    end
end
%}
%intensity_vector = bitflip1k_mex(D,M,initial_vector_seed,intensity_vector);
%intensity_vector = bitflip(D,M,bitflip_count,initial_vector_seed,intensity_vector); % --->
%Accuracy = 85.83 Latin Hypercube
% 786 - 803 - 

for g= 0:high_intensity
    for d = 1:D
        %if (D/M)*(g/D) <= sobol_seq1(d,419)   % Accuracy = 78.97 Kasami, 84.61 Latin Hypercube
        %if (D/M)*(g/D) <= sobol_seq_new(d,1000)
        %if (D/M)*(g/D) <= vd(d,1)
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

classification_percentage = ECT_HDC_Meh_opt_rescale(image_row_size, image_column_size, D, images_train1, images_test1,...
    P_hypervector, intensity_vector, labels_train, labels_test, total_training_images, total_test_images, out_train, out_test )
%end
