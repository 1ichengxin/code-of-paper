function M = fixedrankembeddedfactory(m, n, k)
    M.name = @() sprintf('Manifold of %dx%d matrices of rank %d', m, n, k);
    
    M.dim = @() (m+n-k)*k;
    
    M.inner = @(x, d1, d2) d1.M(:).'*d2.M(:) + d1.Up(:).'*d2.Up(:) ...
                                             + d1.Vp(:).'*d2.Vp(:);
    
    M.norm = @(x, d) sqrt(norm(d.M, 'fro')^2 + norm(d.Up, 'fro')^2 ...
                                             + norm(d.Vp, 'fro')^2);
    
    M.dist = @(x, y) error('fixedrankembeddedfactory.dist not implemented yet.');
    
    M.typicaldist = @() M.dim();
    
    M.tangent = @tangent;
    function Z = tangent(X, Z)
        Z.Up = Z.Up - X.U*(X.U'*Z.Up);
        Z.Vp = Z.Vp - X.V*(X.V'*Z.Vp);
    end


    function ZW = apply_ambient(Z, W)
        if ~isstruct(Z)
            ZW = Z*W;
        else
            ZW = Z.U*(Z.S*(Z.V'*W));
        end
    end

    % Same as apply_ambient, but applies Z' to W.
    function ZtW = apply_ambient_transpose(Z, W)
        if ~isstruct(Z)
            ZtW = Z'*W;
        else
            ZtW = Z.V*(Z.S'*(Z.U'*W));
        end
    end

    M.proj = @projection;
    function Zproj = projection(X, Z)
            
        ZV = apply_ambient(Z, X.V);
        UtZV = X.U'*ZV;
        ZtU = apply_ambient_transpose(Z, X.U);

        Zproj.M = UtZV;
        Zproj.Up = ZV  - X.U*UtZV;
        Zproj.Vp = ZtU - X.V*UtZV';

    end

    M.egrad2rgrad = @projection;
    

    M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(X, egrad, ehess, H)
        
        % Euclidean part
        rhess = projection(X, ehess);
        
        % Curvature part
        T = apply_ambient(egrad, H.Vp)/X.S;
        rhess.Up = rhess.Up + (T - X.U*(X.U'*T));
        T = apply_ambient_transpose(egrad, H.Up)/X.S;
        rhess.Vp = rhess.Vp + (T - X.V*(X.V'*T));
        
    end

    M.tangent2ambient_is_identity = false;
    M.tangent2ambient = @tangent2ambient;
    function Zambient = tangent2ambient(X, Z)
        Zambient.U = [X.U*Z.M + Z.Up, X.U];
        Zambient.S = eye(2*k);
        Zambient.V = [X.V, Z.Vp];
    end
    

    M.retr = @retraction;
    function Y = retraction(X, Z, t)
        if nargin < 3
            t = 1.0;
        end

        [Qu, Ru] = qr([X.U, Z.Up], 0);
        [Qv, Rv] = qr([X.V, Z.Vp], 0);
        

        [U, S, V] = svd(Ru*[X.S + t*Z.M, t*eye(k); t*eye(k), zeros(k)]*Rv');
    
        Y.U = Qu*U(:, 1:k); 
        Y.V = Qv*V(:, 1:k); 
        Y.S = S(1:k, 1:k);
        
        % Equivalent but very slow code
        % [U, S, V] = svds(X.U*X.S*X.V' + t*(X.U*Z.M*X.V' + Z.Up*X.V' + X.U*Z.Vp'), k);
        % Y.U = U; Y.V = V; Y.S = S;
    end

    M.retr_ortho = @retraction_orthographic;
    function Y = retraction_orthographic(X, Z, t)
        if nargin < 3
            t = 1.0;
        end
        
        % First, write Y (the output) as U1*S0*V1', where U1 and V1 are
        % orthogonal matrices and S0 is of size r by r.
        [U1, ~] = qr(t*(X.U*Z.M  + Z.Up) + X.U*X.S, 0);
        [V1, ~] = qr(t*(X.V*Z.M' + Z.Vp) + X.V*X.S, 0);
        S0 = (U1'*X.U)*(X.S + t*Z.M)*(X.V'*V1) ...
                         + t*((U1'*Z.Up)*(X.V'*V1) + (U1'*X.U)*(Z.Vp'*V1));
        
        % Then, obtain the singular value decomposition of Y.
        [U2, S2, V2] = svd(S0);
        Y.U = U1*U2;
        Y.S = S2;
        Y.V = V1*V2;
        
    end



    M.hash = @(X) ['z' hashmd5([sum(X.U(:)) ; sum(X.S(:)); sum(X.V(:)) ])];

    
    M.rand = @random;
    stiefelm = stiefelfactory(m, k);
    stiefeln = stiefelfactory(n, k);
    function X = random()
        X.U = stiefelm.rand();
        X.V = stiefeln.rand();
        X.S = diag(sort(rand(k, 1), 1, 'descend'));
    end

    M.randvec = @randomvec;
    function Z = randomvec(X)
        Z.M  = randn(k);
        Z.Up = randn(m, k);
        Z.Vp = randn(n, k);
        Z = tangent(X, Z);
        nrm = M.norm(X, Z);
        Z.M  = Z.M  / nrm;
        Z.Up = Z.Up / nrm;
        Z.Vp = Z.Vp / nrm;
    end
    
    M.lincomb = @lincomb;
    
    M.zerovec = @(X) struct('M', zeros(k, k), 'Up', zeros(m, k), ...
                                              'Vp', zeros(n, k));
    

    M.transp = @project_tangent;
    function Z2 = project_tangent(X1, X2, Z1)
        Z2 = projection(X2, tangent2ambient(X1, Z1));
    end

    M.vec = @vec;
    function Zvec = vec(X, Z) %#ok<INUSL>
        A = Z.M;
        B = Z.Up;
        C = Z.Vp;
        Zvec = [A(:) ; B(:) ; C(:)];
    end
    rangeM = 1:(k^2);
    rangeUp = (k^2)+(1:(m*k));
    rangeVp = (k^2+m*k)+(1:(n*k));
    M.mat = @(X, Zvec) struct('M',  reshape(Zvec(rangeM),  [k, k]), ...
                              'Up', reshape(Zvec(rangeUp), [m, k]), ...
                              'Vp', reshape(Zvec(rangeVp), [n, k]));
    M.vecmatareisometries = @() true;
    
    

    M.matrix2triplet = @matrix2triplet;
    function X_triplet = matrix2triplet(X_matrix, r)
        if ~exist('r', 'var') || isempty(r) || r <= 0
            r = k;
        end
        if r < min(m, n)
            [U, S, V] = svds(X_matrix, r);
        else
            [U, S, V] = svd(X_matrix, 'econ');
        end
        X_triplet.U = U;
        X_triplet.S = S;
        X_triplet.V = V;
    end
    M.triplet2matrix = @triplet2matrix;
    function X_matrix = triplet2matrix(X_triplet)
        U = X_triplet.U;
        S = X_triplet.S;
        V = X_triplet.V;
        X_matrix = U*S*V';
    end

end

function d = lincomb(x, a1, d1, a2, d2) %#ok<INUSL>

    if nargin == 3
        d.Up = a1*d1.Up;
        d.Vp = a1*d1.Vp;
        d.M  = a1*d1.M;
    elseif nargin == 5
        d.Up = a1*d1.Up + a2*d2.Up;
        d.Vp = a1*d1.Vp + a2*d2.Vp;
        d.M  = a1*d1.M  + a2*d2.M;
    else
        error('fixedrank.lincomb takes either 3 or 5 inputs.');
    end

end
