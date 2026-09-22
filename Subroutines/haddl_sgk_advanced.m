function [D, G] =haddl_sgk_advanced(X, param)
    % dl_sgk_advanced: Advanced SGK algorithm with noise-aware sparse coding and median filtering
    % INPUT
    % X:     input training samples
    % param: parameter struct
    %   param.mode = 1;   % 1: sparsity; 0: error
    %   param.niter = 10; % number of SGK iterations to perform; default: 10
    %   param.D = DCT;    % initial D
    %   param.T = 3;      % sparsity level
    %   param.lambda = 0.1; % L2 regularization parameter
    %   param.alpha = 0.1;  % additional regularization parameter for noise suppression
    % OUTPUT
    % D:    learned dictionary
    % G:    sparse coefficients

    % Extract parameters
    T = param.T;
    niter = param.niter;
    mode = param.mode;
    lambda = param.lambda;
    alpha = param.alpha;

    % Initialize dictionary
    D = param.D;
    [n1, ~] = size(D);
    K = size(D, 2);

    for iter = 1:niter
        % Sparse coding
        if mode == 1
            G = ompN_noise_aware(D, X, T, alpha);
        else
            % Implement error-constrained sparse coding here if needed
        end

        % Adaptive dictionary update with L2 regularization
        for ik = 1:K
            inds = find(G(ik, :) ~= 0);
            if ~isempty(inds)
                D(:, ik) = sum(X(:, inds), 2);
                % Apply L2 regularization to dictionary update
                D(:, ik) = D(:, ik) / norm(D(:, ik));
                % Apply regularization parameter lambda
                D(:, ik) = D(:, ik) / (1 + lambda);
            end
        end
    end

    % Sparse coding with updated dictionary
    G = ompN_noise_aware(D, X, T, alpha);

    % Apply median filtering to sparse coefficients
    G = medfilt2(G, [2, 2]);

    % Return the updated dictionary and sparse coefficients
    return
end

function [G] = ompN_noise_aware(D, X, T, alpha)
    % multi-column sparse coding with noise-aware techniques
    [n1, n2] = size(X);
    [n1, n3] = size(D);
    G = zeros(n3, n2);
    if T == 1
        for i2 = 1:n2
            G(:, i2) = omp_e_noise_aware(D, X(:, i2), alpha);
        end
    else
        for i2 = 1:n2
            G(:, i2) = dl_omp0_noise_aware(D, X(:, i2), T, alpha);
        end
    end
    return
end

function [g] = omp_e_noise_aware(D, x, alpha)
    % Basic orthogonal matching pursuit for sparse coding with noise awareness
    [n1, n2] = size(D);
    g = zeros(n2, 1);

    max_val = 0;
    for i2 = 1:n2
        dtr = abs(sum(D(:, i2) .* x)) - alpha * norm(x - D(:, i2) * g(i2));
        if max_val < dtr
            max_val = dtr;
            k = i2;
        end
    end

    g(k) = sum(D(:, k) .* x) / sum(D(:, k) .* D(:, k));

    return
end

function [g] = dl_omp0_noise_aware(D, x, K, alpha)
    % Orthogonal matching pursuit with noise awareness for sparse coding
    [n1, n2] = size(D);
    I = [];
    r = x;
    g = zeros(n2, 1);
    for ik = 1:K
        k = [];
        max_val = 0;
        for i2 = 1:n2
            if sum(find(I == i2)) == 0
                dtr = abs(sum(D(:, i2) .* r)) - alpha * norm(r - D(:, i2) * g(i2));
                if max_val < dtr
                    max_val = dtr;
                    k = i2;
                end
            end
        end
        I = [I, k];
        g(I) = inv(D(:, I)' * D(:, I)) * D(:, I)' * x;
        r = x - D(:, I) * g(I);
    end

    return
end
