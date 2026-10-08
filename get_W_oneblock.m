function W = get_W_oneblock(N0, eps, g2total, conserved, type)
    N = N0 * 2; % Total number of neurons
    if conserved
        g = sqrt(g2total-eps^2)/2; % variance conserved
    else
        g = sqrt(g2total)/2; % variance not conserved
    end
    W0 = randn(N)*g / sqrt(N);

    if type == "bottom right"
        W_eps = [zeros(N0), zeros(N0); zeros(N0), randn(N0)*eps] / sqrt(N); % bottom right block
    elseif type == "top left"
        W_eps = [randn(N0)*eps, zeros(N0); zeros(N0), zeros(N0)] / sqrt(N); % top left block
    elseif type == "top right"
        W_eps = [zeros(N0), randn(N0)*eps; zeros(N0), zeros(N0)] / sqrt(N); % top right block
    else 
        W_eps = [zeros(N0), zeros(N0); randn(N0)*eps, zeros(N0)] / sqrt(N); % bottom left block
    end
    
    W = W0 + W_eps;
end
