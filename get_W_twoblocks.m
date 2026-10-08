function W = get_W_twoblocks(N0, eps, g2total, conserved, type)
    N = N0 * 2; % Total number of neurons
    
    if conserved
        g = sqrt(g2total-eps^2)/2; % variance conserved
    else
        g = sqrt(g2total)/2; % variance not conserved
    end
    W0 = randn(N)*g / sqrt(N);

    if type == "diagonals"
        W_eps = [randn(N0)*eps/sqrt(2), zeros(N0); zeros(N0), randn(N0)*eps/sqrt(2)] / sqrt(N); % diagonal blocks
    elseif type == "off diagonals"
        W_eps = [zeros(N0), randn(N0)*eps/sqrt(2); randn(N0)*eps/sqrt(2), zeros(N0)] / sqrt(N); % off diagonals
    elseif type == "right blocks"
        W_eps = [zeros(N0), randn(N0)*eps/sqrt(2); zeros(N0), randn(N0)*eps/sqrt(2)] / sqrt(N); % right blocks
    else 
        W_eps = [zeros(N0), zeros(N0); randn(N0)*eps/sqrt(2), randn(N0)*eps/sqrt(2)] / sqrt(N); % bottom blocksk
    end
    W = W0 + W_eps;
end