function D = createDCTMatrix(c1,c2)

    % Initialize the DCT matrix
    dct = zeros(c1, c2);

    % Compute the DCT basis vectors
    for k = 0:c2-1
        % Create the cosine vector for the k-th frequency
        V = cos((0:c1-1)' * k * pi / c2);

        % Remove the mean for k > 0 to ensure orthogonality
        if k > 0
            V = V - mean(V);
        end

        % Normalize the vector
        dct(:, k+1) = V / norm(V);
    end

    % Output the DCT matrix
    D = dct;
end
