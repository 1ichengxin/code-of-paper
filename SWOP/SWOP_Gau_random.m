function SWOP_Gau_random()

    m = 400;
    n = 400;
    k = 30;
    % Generate a random mxn matrix A of rank k
    L = randn(m, k);
    R = randn(n, k);
    A = L*R';
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %接下来我们将数据限制在 (0,1)之内 (value - vmin)/(vmax-vmin).
    minvalue = min(A,[],'all');
    maxvalue = max(A,[],'all');
    A = (A-minvalue)/(maxvalue-minvalue);
    size(A)
    epsilon = 3;
    e = 0.81/epsilon;
    N = randn(m,n)*e;

    % Generate a random mask for observed entries: P(i, j) = 1 if the entry
    % (i, j) of A is observed, and 0 otherwise.
    fraction = 0.6;
    P = sparse(rand(m, n) <= fraction);
    % Hence, we know the nonzero entries in PA:
    A = A + 0.4*N;
    PA = P.*A;
    Num = nnz(PA);

    problem.M = fixedrankembeddedfactory(m, n, k);

    problem.cost = @cost;
    function f = cost(X)
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
    %fprintf('||NP*(X-A)||_F/UNum = %g\n', norm(NP.*(Xmat - A), 'fro')/sqrt(UNum));
    fprintf('SE = %g\n', norm(P.*(Xmat - A), 'fro')/sqrt(Num));
    fprintf('RE = %g\n', norm(P.*(Xmat - A), 'fro')/norm(PA, "fro"));


end