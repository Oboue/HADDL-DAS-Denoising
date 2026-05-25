function [D, G] = conv_dl_sgk(X, param)
    % conv_dl_sgk: Convolutional SGK algorithm
    % INPUT
    % X:     input training sample (2D array, e.g., an image)
    % param: parameter struct
    %   param.mode = 1;   % 1: sparsity; 0: error (not implemented)
    %   param.niter = 10; % number of SGK iterations to perform; default: 10
    %   param.D = DCT;    % initial D (convolutional filters)
    %   param.T = 3;      % sparsity level
    % OUTPUT
    % D: learned convolutional dictionary (filters)
    % G: sparse coefficients

    T = param.T;
    niter = param.niter;
    mode = param.mode;
    D = param.D;

    for iter = 1:niter
        if mode == 1
            G = ompN_conv(D, X, T);
        else
            % Error-defined sparse coding (not implemented)
            error('Error-defined sparse coding not implemented');
        end

        for ik = 1:size(D, 3) % Update each filter
            g_slice = squeeze(G(ik, :, :));
            inds = find(g_slice ~= 0);
            if ~isempty(inds)
                D_update = zeros(size(D, 1), size(D, 2));
                for ind = inds'
                    [row, col] = ind2sub(size(g_slice), ind);
                    patch = X(row:row+size(D, 1)-1, col:col+size(D, 2)-1);
                    D_update = D_update + patch * g_slice(row, col);
                end
                D(:, :, ik) = D_update / norm(D_update, 'fro');
            end
        end
    end

    % Extra step
    G = ompN_conv(D, X, T);
end

function [G] = ompN_conv(D, X, K)
    % Multi-column sparse coding for convolutional dictionary

    [n1, n2] = size(X);
    [n1_d, n2_d, n3_d] = size(D);
    G = zeros(n3_d, n1 - n1_d + 1, n2 - n2_d + 1);

    if K == 1
        G = omp_conv(D, X);
    else
        G = dl_omp_conv(D, X, K);
    end
end

function [g] = omp_conv(D, x)
    % Convolutional OMP for sparse coding

    [n1_d, n2_d, n3_d] = size(D);
    g = zeros(n3_d, size(x, 1) - n1_d + 1, size(x, 2) - n2_d + 1);

    max_val = 0;
    best_filter = 0;
    row = 0;
    col = 0;
    for i3 = 1:n3_d
        conv_res = conv2(x, rot90(D(:, :, i3), 2), 'valid');
        [max_conv, max_idx] = max(abs(conv_res(:)));
        if max_conv > max_val
            max_val = max_conv;
            best_filter = i3;
            [row, col] = ind2sub(size(conv_res), max_idx);
        end
    end

    if best_filter > 0
        conv_result = conv2(x, rot90(D(:, :, best_filter), 2), 'valid');
        g(best_filter, row, col) = conv_result(row, col);
    end
end

function [g] = dl_omp_conv(D, x, K)
    % Convolutional OMP for sparse coding with sparsity level K

    [n1_d, n2_d, n3_d] = size(D);
    g = zeros(n3_d, size(x, 1) - n1_d + 1, size(x, 2) - n2_d + 1);

    I = [];
    r = x;

    for ik = 1:K
        max_val = 0;
        best_filter = 0;
        row = 0;
        col = 0;
        for i3 = 1:n3_d
            if isempty(I) || ~ismember(i3, I)
                conv_res = conv2(r, rot90(D(:, :, i3), 2), 'valid');
                [max_conv, max_idx] = max(abs(conv_res(:)));
                if max_conv > max_val
                    max_val = max_conv;
                    best_filter = i3;
                    [row, col] = ind2sub(size(conv_res), max_idx);
                end
            end
        end

        if best_filter > 0
            I = [I, best_filter];
            conv_result = conv2(x, rot90(D(:, :, best_filter), 2), 'valid');
            g(best_filter, row, col) = conv_result;
            r = r - conv2(g(best_filter, row, col), D(:, :, best_filter), 'full');
        end
    end
end
