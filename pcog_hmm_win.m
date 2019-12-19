function [whmm] = pcog_hmm_win(hmm,o,n,gen)
% [whmm] = pcog_hmm_win(hmm,o,n,gen)
% Inference and learning for HMMs using sliding window
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
% n - window length
% gen - Generative model (optional)
%   gen.A - transition matrix
%   gen.B - emission matrix
%   gen.d - initial state vector 
%   gen.x - true hidden states
%
% Outputs
% whmm.x - Final state estimates
% whmm.xf - Online (filtered) state estimates
% whmm.xp - Online predictions
% whmm.op - Predicted observations (online)
% whmm.pA - Concentration parameters for transition matrix
% whmm.pB - Concentration parameters for emission matrix
% whmm.pd - Concentration parameters for initial state vector
%
% TF 08/17

if nargin<4, gen = []; end

if ~isfield(hmm,'special'), hmm.special = 'No'; end
w.special = hmm.special;

no = size(o,2);
ns = size(hmm.pA,2);
% Priors
%--------------------------------------------------------------------------
% uA = hmm.pA;
% uB = hmm.pB;
ud = hmm.pd;

pA = hmm.pA;
pB = hmm.pB;
pd = hmm.pd;

w.nit = hmm.nit;
w.tol = hmm.tol;

% Memory mapping
%--------------------------------------------------------------------------
x = zeros(ns,no); xf=x;
xp = zeros(ns,no+1); xp(:,1) = ud'./sum(ud);
op = xp; 
op(:,1) =(exp(psi(pB)-repmat(psi(sum(pB,2)),1,ns)))'*xp(:,1); %Predictions about first observation
op(:,1) = op(:,1)./sum(op(:,1));
tpA = zeros(ns,ns,no); tpB = tpA; Ahat = tpA; Bhat = tpA;
tpA_L = tpA; tpB_L = tpA; Ahat_L = tpA; Bhat_L = tpA;
dhat = zeros(ns,no);

for i=1:no
     
     % Find how long the window should be and select observations
     %---------------------------------------------------------------------
     dostep = min([n i]); % What the window length should be
     tx = i-dostep+1:i; % Which observations should be used 
     ow = o(:,tx);  % Select observations in window
     
     % Initialise priors
     %---------------------------------------------------------------------
     w.pA = pA; % Transition matrix
     w.pB = pB; % Observation matrix
     if i<=n %Less than a single window
        w.pd = ud; % Initial state distribution (d)
     else 
        w.pd = 1e8*xf(:,tx(1)-1)'; % Treat last state outside time window as fixed
     end
     
     %Inference over the window
     %---------------------------------------------------------------------
     [w] = pcog_hmm_VB(w,ow,tx(1)>1); 
     
      % Update parameter estimates
      %--------------------------------------------------------------------
     if strcmp(hmm.special,'No') % Standard HMM 
         if i>=n
            pB = pB + w.x(:,1)*ow(:,1)'; %Update emission matrix (B)
            if i==n
                pd = pd + x(:,1)'; % Update initial state (d)
            else pA = pA + w.xi(:,:,1);
            end
         end
     elseif strcmp(hmm.special,'rev') % Reversal 
         if i>=n
            pB = pB + eye(2)*sum(diag(w.x(:,1)*ow(:,1)')) + rot90((eye(2)*sum(diag(rot90(w.x(:,1)*ow(:,1)'))))); %Enforce symmetry
            if i==n
                pd = pd + x(:,1)'; % Update initial state (d)
            else pA = pA + eye(2)*sum(diag(w.xi(:,:,1))) + rot90((eye(2)*sum(diag(rot90(w.xi(:,:,1))))));
            end
         end
     end
     
     % Store state estimates
     %---------------------------------------------------------------------
     xf(:,i) = w.x(:,end); % Online state estimate
     xp(:,i+1) = (w.Ahat'*w.x(:,end))/sum(w.Ahat'*w.x(:,end)); %Online prediction of next state
     x(:,tx) = w.x; % Offline state estimate (final)
     op(:,i+1) = (w.Bhat'*xp(:,i+1))/sum(w.Bhat'*xp(:,i+1)); %Prediction about next observation
     
     % Store parameter estimates
     %---------------------------------------------------------------------
     if i==1
         w.pA = w.uA; % No learning about transitions on first trial
     end
     tpA(:,:,i) = w.pA;
     tpB(:,:,i) = w.pB;
     Ahat(:,:,i) = exp(psi(w.pA)-repmat(psi(sum(w.pA,2)),1,ns)); %Possibly could include a random element here to avoid local minima
     Bhat(:,:,i) = exp(psi(w.pB)-repmat(psi(sum(w.pB,2)),1,ns));
     dhat(:,i) = exp(psi(pd)-(psi(sum(pd,2))));
     % Lagged estimates
     tpA_L(:,:,i) = pA;
     tpB_L(:,:,i) = pB;
     Ahat_L(:,:,i) = exp(psi(pA)-repmat(psi(sum(pA,2)),1,ns)); %Possibly could include a random element here to avoid local minima
     Bhat_L(:,:,i) = exp(psi(pB)-repmat(psi(sum(pB,2)),1,ns));
     
     % Convergence properties
     %---------------------------------------------------------------------
     jit(i) = w.jit;
     conv(i) = w.conv;
     
end

whmm.x = x;
whmm.xf = xf;
whmm.xp = xp(:,1:no);
whmm.op = op(:,1:no);
whmm.pA = tpA;
whmm.pB = tpB;
whmm.pd = pd;
whmm.Ahat = Ahat;
whmm.Bhat = Bhat;
whmm.dhat = dhat;
whmm.pA_L = tpA_L;
whmm.pB_L = tpB_L;
whmm.Ahat_L = Ahat_L;
whmm.Bhat_L = Bhat_L;
whmm.uA = hmm.pA;
whmm.uB = hmm.pB;
whmm.ud = hmm.pd;
whmm.jit = jit;
whmm.conv = conv;
whmm.special = hmm.special;

% Calculate accuracy
%--------------------------------------------------------------------------
if ~isempty(gen)
    [whmm] = pcog_hmm_VB_acc(whmm,o,gen.x,gen.A,gen.B,gen.d);
end 
        
        
        
        
    
    
    


