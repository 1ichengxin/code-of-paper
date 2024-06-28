function [x, cost, info, options] = conjugategradient(problem, x, options)

M = problem.M;

if ~canGetCost(problem)
    warning('manopt:getCost', ...
        'No cost provided. The algorithm will likely abort.');
end
if ~canGetGradient(problem) && ~canGetApproxGradient(problem)
    warning('manopt:getGradient:approx', ...
           ['No gradient provided. Using an FD approximation instead (slow).\n' ...
            'It may be necessary to increase options.tolgradnorm.\n' ...
            'To disable this warning: warning(''off'', ''manopt:getGradient:approx'')']);
    problem.approxgrad = approxgradientFD(problem);
end

localdefaults.minstepsize = 1e-8;
localdefaults.maxiter =1000;
localdefaults.tolgradnorm =1e-8;
localdefaults.storedepth = 20;

localdefaults.beta_type = 'P-R';
localdefaults.orth_value = Inf; % by BM as suggested in Nocedal and Wright


if ~canGetLinesearch(problem)
    localdefaults.linesearch = @linesearch_adaptive;
else
    localdefaults.linesearch = @linesearch_hint;
end

localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
if ~exist('options', 'var') || isempty(options)
    options = struct();
end
options = mergeOptions(localdefaults, options);

timetic = tic();

if ~exist('x', 'var') || isempty(x)
    x = M.rand();
end

storedb = StoreDB(options.storedepth);
key = storedb.getNewKey();

[cost, grad] = getCostGrad(problem, x, storedb, key);
gradnorm = M.norm(x, grad);
Pgrad = getPrecon(problem, x, grad, storedb, key);
gradPgrad = M.inner(x, grad, Pgrad);


iter = 0;

stats = savestats();
info(1) = stats;
info(min(10000, options.maxiter+1)).iter = [];


if options.verbosity >= 2
    fprintf(' iter\t               cost val\t    grad. norm\n');
end

desc_dir = M.lincomb(x, -1, Pgrad);


while true
    
    % Display iteration information
    if options.verbosity >= 2
        %fprintf('%5d\t%+.16e\t%.8e\n', iter, cost, gradnorm);
    end
    
    timetic = tic();
    
    [stop, reason] = stoppingcriterion(problem, x, options, info, iter+1);
    
    if ~stop && abs(stats.stepsize) < options.minstepsize
        stop = true;
        reason = sprintf(['Last stepsize smaller than minimum '  ...
                          'allowed; options.minstepsize = %g.'], ...
                          options.minstepsize);
    end
    
    if stop
        if options.verbosity >= 1
            fprintf([reason '\n']);
        end
        break;
    end
    
    
    df0 = M.inner(x, grad, desc_dir);
        
    if df0 >= 0
        
        % Or we switch to the negative gradient direction.
        if options.verbosity >= 3
            fprintf(['Conjugate gradient info: got an ascent direction '...
                     '(df0 = %2e), reset to the (preconditioned) '...
                     'steepest descent direction.\n'], df0);
        end
        % Reset to negative gradient: this discards the CG memory.
        desc_dir = M.lincomb(x, -1, Pgrad);
        df0 = -gradPgrad;
        
    end
    
    
    [stepsize, newx, newkey, lsstats] = options.linesearch( ...
                   problem, x, desc_dir, cost, df0, options, storedb, key);
               
    
    [newcost, newgrad] = getCostGrad(problem, newx, storedb, newkey);
    newgradnorm = M.norm(newx, newgrad);
    Pnewgrad = getPrecon(problem, newx, newgrad, storedb, newkey);
    newgradPnewgrad = M.inner(newx, newgrad, Pnewgrad);
    
    
    if strcmpi(options.beta_type, 'steep') || ...
       strcmpi(options.beta_type, 'S-D')              % Gradient Descent
        
        beta = 0;
        desc_dir = M.lincomb(newx, -1, Pnewgrad);
        
    else
        
        oldgrad = M.transp(x, newx, grad);
        orth_grads = M.inner(newx, oldgrad, Pnewgrad) / newgradPnewgrad;
        
        % Powell's restart strategy (see page 12 of Hager and Zhang's
        % survey on conjugate gradient methods, for example)
        if abs(orth_grads) >= options.orth_value
            beta = 0;
            desc_dir = M.lincomb(x, -1, Pnewgrad);
            
        else % Compute the CG modification
            
            desc_dir = M.transp(x, newx, desc_dir);
            
            switch upper(options.beta_type)
            
                case 'F-R'  % Fletcher-Reeves
                    beta = newgradPnewgrad / gradPgrad;
                
                case 'P-R'  % Polak-Ribiere+
                    % vector grad(new) - transported grad(current)
                    diff = M.lincomb(newx, 1, newgrad, -1, oldgrad);
                    ip_diff = M.inner(newx, Pnewgrad, diff);
                    beta = ip_diff / gradPgrad;
                    beta = max(0, beta);
                
                case 'H-S'  % Hestenes-Stiefel+
                    diff = M.lincomb(newx, 1, newgrad, -1, oldgrad);
                    ip_diff = M.inner(newx, Pnewgrad, diff);
                    beta = ip_diff / M.inner(newx, diff, desc_dir);
                    beta = max(0, beta);

                case 'H-Z' % Hager-Zhang+
                    diff = M.lincomb(newx, 1, newgrad, -1, oldgrad);
                    Poldgrad = M.transp(x, newx, Pgrad);
                    Pdiff = M.lincomb(newx, 1, Pnewgrad, -1, Poldgrad);
                    deno = M.inner(newx, diff, desc_dir);
                    numo = M.inner(newx, diff, Pnewgrad);
                    numo = numo - 2*M.inner(newx, diff, Pdiff)*...
                                     M.inner(newx, desc_dir, newgrad) / deno;
                    beta = numo / deno;

                    % Robustness (see Hager-Zhang paper mentioned above)
                    desc_dir_norm = M.norm(newx, desc_dir);
                    eta_HZ = -1 / ( desc_dir_norm * min(0.01, gradnorm) );
                    beta = max(beta, eta_HZ);
                
                case 'L-S' % Liu-Storey+ from Sato
                    diff = M.lincomb(newx, 1, newgrad, -1, oldgrad);
                    ip_diff = M.inner(newx, Pnewgrad, diff);
                    denom = -1*M.inner(x, grad, desc_dir);
                    betaLS = ip_diff / denom;
                    betaCD = newgradPnewgrad / denom;
                    beta = max(0, min(betaLS, betaCD));

                otherwise
                    error(['Unknown options.beta_type. ' ...
                           'Should be steep, S-D, F-R, P-R, H-S, H-Z, or L-S.']);
            end
            
            desc_dir = M.lincomb(newx, -1, Pnewgrad, beta, desc_dir);
        
        end
        
    end
    
    storedb.removefirstifdifferent(key, newkey);
    x = newx;
    key = newkey;
    cost = newcost;
    grad = newgrad;
    Pgrad = Pnewgrad;
    gradnorm = newgradnorm;
    gradPgrad = newgradPnewgrad;
    
    iter = iter + 1;
    
    storedb.purge();
    
    % Log statistics for freshly executed iteration.
    stats = savestats();
    info(iter+1) = stats;
    
end


info = info(1:iter+1);

if options.verbosity >= 1
    fprintf('Total time is %f [s] (excludes statsfun)\n', info(end).time);
end


% Routine in charge of collecting the current iteration stats
function stats = savestats()
    stats.iter = iter;
    stats.cost = cost;
    stats.gradnorm = gradnorm;
    if iter == 0
        stats.stepsize = nan;
        stats.time = toc(timetic);
        stats.linesearch = [];
        stats.beta = 0;
    else
        stats.stepsize = stepsize;
        stats.time = info(iter).time + toc(timetic);
        stats.linesearch = lsstats;
        stats.beta = beta;
    end
    stats = applyStatsfun(problem, x, storedb, key, options, stats);
end

end


