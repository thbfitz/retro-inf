function [KL] = pcog_DirKL(Q,P)
%function [KL] = pcog_DirKL(Q,P)
% KL Divergence between two Dirichlet distributions
% KL(Q||P)
% TF 08/14


% KL(1) = log(exp(gammaln(sum(Q)))/exp(gammaln(sum(P))));
% KL(2) = sum(log ( (exp(gammaln(P))) ./ (exp(gammaln(Q))) ) );
% KL(3) = sum( (Q - P).*(psi(Q) - psi(sum(Q))) );

KL(1) = (gammaln(sum(Q)) - gammaln(sum(P)));
KL(2) = sum(gammaln(P) - gammaln(Q) );
KL(3) = sum( (Q - P).*(psi(Q) - psi(sum(Q))) );

KL = sum(KL);