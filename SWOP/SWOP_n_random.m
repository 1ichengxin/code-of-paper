function SWOP_n_random()

    m = 400;
    n = 400;
    k = 30;
    % Generate a random mxn matrix A of rank k
    L = randn(m, k);
    R = randn(n, k);
    A = L*R';
    % Generate a random mask for observed entries: P(i, j) = 1 if the entry
    % (i, j) of A is observed, and 0 otherwise.
    %fraction = 2 * k*(m+n-k)/(m*n);
    fraction = 0.2;
    P = sparse(rand(m, n) <= fraction);
%     NP = ~P;
%     UNum = nnz(NP);
    % Hence, we know the nonzero entries in PA:
    PA = P.*A;
    Num = nnz(PA);


    problem.M = fixedrankembeddedfactory(m, n, k);


    problem.cost = @cost;
    function f = cost(X)
        % Note that it is very much inefficient to explicitly construct the
        % matrix X in this way. Seen as we only need to know the entries
        % of Xmat corresponding to the mask P, it would be far more
        % efficient to compute those only.
        Xmat = X.U*X.S*X.V';
        f = .5*norm( P.*Xmat - PA , 'fro')^2;
    end


    problem.egrad = @egrad;
    function G = egrad(X)
        % Same comment here about Xmat.
        Xmat = X.U*X.S*X.V';
        G = P.*Xmat - PA;
    end


    [U, S, V] = svds(PA, k);
    X0.U = U;
    X0.S = S;
    X0.V = V;

    [X, xcost, info, options] = conjugategradient(problem, X0); %#ok<ASGLU>

    Xmat = X.U*X.S*X.V';
    fprintf('SE = %g\n', norm(P.*(Xmat - A), 'fro')/sqrt(Num));
    fprintf('RE = %g\n', norm(P.*(Xmat - A), 'fro')/norm(P.*A,'fro'));
    %fprintf('EE = %g\n', norm(Xmat - A, 'fro')/norm(A,'fro'));
end



    