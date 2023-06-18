function out_shape = shapedim18(image_in)
    out_shape = zeros(18, 18);
    for ii = 0:17
        out_shape(:,ii+1) = image_in((ii*18)+1:(ii*18)+18)';
        %k = k + 28;
    end
    %out_shape = imresize(out_shape, [16 16]);
end