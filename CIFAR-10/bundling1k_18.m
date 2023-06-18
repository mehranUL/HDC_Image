function bundled = bundling8k_18(image_row_size, image_column_size,D, xored_Images)

    bundled = zeros(1, D);
    reshaped_xored_Images = zeros(1, D);

    %for item_image=1:1:trainDB_size %iterating over training images
        for i=1:1:image_row_size
            for j = 1:1:image_column_size
                
                reshaped_xored_Images(1,:) = xored_Images(i,j,:);
                %bundled(item_image, :) = bundled(item_image, :) + reshaped_xored_Images(item_image, :);
                bundled(1, :) = bundled(1, :) + reshaped_xored_Images(1, :);
            end
        end
end