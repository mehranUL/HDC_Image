function xored_Images1 = binding1k_18(image_row_size,image_column_size,D,P_hypervector,intensity_vector,shaped_images)
    xored_Images1 = zeros(image_row_size,image_column_size, D);
    reshaped_P_hv = zeros(1, D);
    for i=1:1:image_row_size
            %shaped_images(1,image_row_size,:) = images_train(k:k+27, TRAIN_IMAGE_INDEX);
        for j = 1:1:image_column_size
                %for meh = 1:D
                 %   reshaped_P_hv(item_image, meh) = P_hypervector(i, j, meh);
                %end
                reshaped_P_hv(1,:) = P_hypervector(i,j,:);
                xored_Images1(i, j, :) = reshaped_P_hv(1,:) .* intensity_vector(shaped_images(i,j)+1,:);
                
        end
    end
end