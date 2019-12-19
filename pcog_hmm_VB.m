function [hmm] = pcog_hmm_VB(hmm,o,wmode)
% function [hmm] = pcog_hmm_VB(hmm,o,wmode)
% Perform infernece and learning for an HMM using variational EM (see Beal
% 2003)
%
% Inputs
% hmm.pA - Concentration parameters for transition matrix
% hmm.pB - Concentration parameters for emission matrix
% hmm.pd - Concentration parameters for initial state vector
% hmm.nit -  Max iterations for VBEM
% hmm.tol -  Tolerance per data point for convergence
% hmm.special - Using a special case? If unspecified, then defaults to standard.
%               Possible values: 'rev' - reversal learning 
% o - observations
% wmode - in a window or not (0 when first time point in window is first observation)
%
% Outputs
% hmm.pA - Updated concentration parameters for transition matrix
% hmm.pB - Updated concentration parameters for emission matrix
% hmm.pd - Updated concentration parameters for initial state vector
% hmm.uA - Initial concentration parameters for transition matrix
% hmm.uB - Initial concentration parameters for emission matrix
% hmm.ud - Initial concentration parameters for initial state vector
% hmm.x - State estimates (gamma)
% hmm.xf - Forward inference (normalised alpha)
% hmm.xb - Backward inference (normalised beta)
% hmm.scale - Normalisation constants (Z)
% hmm.xi - Dual-slice marginals
% hmm.F - Variational lower bound
% hmm.Ahat - Subnormalised transition matrix
% hmm.Bhat - Subnormalised emission matrix
% hmm.dhat - Subnormalised initial state vector
% hmm.A - 'True' transition matrices
% hmm.B - 'True' emission matrices
% hmm.d - 'True' initial state vector
% hmm.jit - Number of iterations before convergence
% hmm.conv - Did convergence occur? (0/1)
%
% TF 08/17

if ~isfield(hmm,'special'), hmm.special = 'No'; end

% Housekeeping
nit = hmm.nit; %Max number of iterations
ns = size(hmm.pA,2); % Number of hidden states
no = size(o,2); % Length of data sequence
tol = hmm.tol; % Tolerance for convergence 
ntol = no*tol; % Adjust tolerance to data length

% Priors
uA = hmm.pA;
uB = hmm.pB;
ud = hmm.pd;

% Initialise counts
cA = zeros(size(uA));
cB = zeros(size(uB));
cd = zeros(size(ud));
%Initialise lower bound
F = zeros(5,nit);

for jit = 1:nit
    
    % M step (parameter learning)
    pA = uA + cA;
    pB = uB + cB;
    pd = ud + cd;
    
    Ahat = exp(psi(pA)-repmat(psi(sum(pA,2)),1,ns)); %Possibly could include a random element here to avoid local minima
    Bhat = exp(psi(pB)-repmat(psi(sum(pB,2)),1,ns));
    dhat = exp(psi(pd)-(psi(sum(pd,2))));
    
    % E step
    [x,xf,xb,scale,xi] = pcog_hmm_FB(Ahat,Bhat,dhat',o,wmode); % Smoothing
    
     % Compute F, straight after E Step.
     F(1,jit) = sum(log(scale));
     if strcmp(hmm.special,'No') % Standard HMM
         for js = 1:ns
              F(2,jit) =  F(2,jit) - pcog_DirKL(pA(js,:),uA(js,:));
              F(3,jit) =  F(3,jit) - pcog_DirKL(pB(js,:),uB(js,:));
         end
     elseif strcmp(hmm.special,'rev') % Reversal learning
         for js = 1%:ns
            F(2,jit) =  F(2,jit) - pcog_DirKL(pA(js,:),uA(js,:));
            F(3,jit) =  F(3,jit) - pcog_DirKL(pB(js,:),uB(js,:));
        end 
     end
     F(4,jit) =  F(4,jit) - pcog_DirKL(pd,ud);
     F(5,jit) =  sum(F([1:4],jit));
  
    % Counts
    if strcmp(hmm.special,'No') % Standard HMM
        cA = sum(xi,3);
        cB = x*o'; % Check this
    elseif strcmp(hmm.special,'rev')
        cA = eye(2)*sum(diag(sum(xi,3))) + rot90((eye(2)*sum(diag(rot90(sum(xi,3))))));
        cB = eye(2)*sum(diag(x*o')) + rot90((eye(2)*sum(diag(rot90(x*o'))))); 
    end
    cd = x(:,1)'; %Double check this
  
  % Check for convergence or errors  
  if (jit>2) 
    if (F(5,jit)<(F(5,jit-1) - 1e-6))     fprintf('violation');
    elseif (F(5,jit)-F(5,jit-1))<ntol
        conv = 1;
        break    
    elseif jit==nit
        fprintf('failed to converge')
        conv = 0;
    end
  end
        
end

hmm.pA = pA;
hmm.pB = pB;
hmm.pd = pd;
hmm.uA = uA;
hmm.uB = uB;
hmm.ud = ud;
hmm.x = x;
hmm.xf = xf;
hmm.xb = xb;
hmm.scale = scale;
hmm.xi = xi;
hmm.F = F(:,1:jit);
hmm.Ahat = Ahat; % Subnormalised transition matrices
hmm.Bhat = Bhat;
hmm.dhat = dhat;
hmm.A = pA./repmat(sum(pA,2),1,ns); % 'True' transition matrices
hmm.B = pB./repmat(sum(pB,2),1,ns);
hmm.d = pd./repmat(sum(pd,2),1,ns);
hmm.jit = jit; %Number of iterations before convergence
hmm.conv = conv;
% hmm.special = special;



    



