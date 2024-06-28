function Copy_of_SWOP_random()
    m = 400;
    n = 400;                     %Data size
    k = 30;                      %Rank
    L = randn(m,k);
    R = randn(n,k);
    A = L * R';    %生成初始的随机矩阵 (400,400,30)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %接下来我们将数据限制在 (0,1)之内 (value - vmin)/(vmax-vmin).
    minvalue = min(A,[],'all');
    maxvalue = max(A,[],'all');
    A = (A-minvalue)/(maxvalue-minvalue);
    size(A)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %首先确定采样率，对于\epsilon的变化，我们先设计采样率为0.6.
    %对于需要确定采样率的变化曲线，我们设计采样率为 0.1,0.3,0.5,0.7,0.9
    Ind = randperm(m*n);   %打乱顺序，目前是实现均匀随机取样
    rate = 0.4;            % 采样率设计为 0.6
    p1 = floor(m*n*rate);  % 舍人到小于等于的整数
    P = ones(m,n);
    P(Ind(1:p1)) = 0;     % 采样单元矩阵
    NP = ones(m,n)-P;
    UNum = nnz(NP)
    SA = P.*A;            % 采样数据矩阵
    [i,j,v] = find(SA);
    num = length(v)
    disp(['the length of v is: ', num2str(num)])
    disp(['the length of i is: ', num2str(length(i))])
    disp(['the length of j is: ', num2str(length(j))])
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %接下来我们开始利用截断高斯进行采样操作，
    TGMX_Noi = zeros(num,1);
    noiseset = zeros(num,1);
    epsilons = 0.7;         %我们首先看当epsilon=1的情况 
    for c = 1:num
        b = 0.81/epsilons;
        while true
            randvalue = randn(1)*b;
            value = v(c) + randvalue;  
            if value > 0 && value < 1
                TGMX_Noi(c) = value;
                noiseset(c) = randvalue;
                break
            end
        end
    end
    Sparse_TGMX_Noi = sparse(i,j,noiseset);
    N = full(Sparse_TGMX_Noi);
    PA = SA+0.4*N;
    size(PA)
    Num = nnz(PA)


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
    fprintf('SE = %g\n', norm(P.*(Xmat - A), 'fro')/sqrt(Num));
    fprintf('RE = %g\n', norm(P.*(Xmat - A), 'fro')/norm(P.*A,'fro'));
    %fprintf('EE = %g\n', norm((Xmat - A), 'fro')/norm(A,'fro'));


end