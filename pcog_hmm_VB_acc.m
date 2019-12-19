function [hmm] = pcog_hmm_VB_acc(hmm,o,x,A,B,d)
% [hmm] = pcog_hmm_VB_acc(hmm,o,x,A,B,d)
% Test accuracy of hmmVB simulations


if ~isfield(hmm,'special'), hmm.special = 'No'; end

LL.x = sum(sum(log(hmm.x).*x)); % Offline (retrospective) belief accuracy 
LL.xf = sum(sum(log(hmm.xf).*x)); % Online belief accuracy
try, LL.xp = sum(sum(log(hmm.xp).*x)); end % Onlineprediction accuracy
try, LL.op = sum(sum(log(hmm.op).*o)); end %Online prediction of observations

LL.pA = zeros(1,size(hmm.pA,3));
LL.pB = LL.pA;% LL.oA = LL.pA;
SS.A = LL.pA;
for t = 1:size(hmm.pA,3)
    if strcmpi(hmm.special,'No') % Standard HMM
        for j=1:size(A,1)
            LL.pA(t) = LL.pA(t) + log(spm_Dpdf(A(j,:),hmm.pA(j,:,t)));
            LL.pB(t) = LL.pB(t) + log(spm_Dpdf(B(j,:),hmm.pB(j,:,t)));
            SS.A(t) = SS.A(t) + sum((A(j,:)-(hmm.pA(j,:,t)./sum(hmm.pA(j,:,t)))).^2);
        end
    elseif strcmpi(hmm.special,'rev') % Reversal learning
        for j=1%:size(A,1)
            LL.pA(t) = LL.pA(t) + log(spm_Dpdf(A(j,:),hmm.pA(j,:,t)));
            LL.pB(t) = LL.pB(t) + log(spm_Dpdf(B(j,:),hmm.pB(j,:,t)));
            SS.A(t) = SS.A(t) + sum((A(j,:)-(hmm.pA(j,:,t)./sum(hmm.pA(j,:,t)))).^2);
        end
    end
end
% % Alternative
% LL.pA2 = zeros(1,size(hmm.pA,3));
% LL.pB2 = LL.pA2;% LL.oA = LL.pA;
% for t = 1:size(hmm.pA,3)
%     for j=1:size(A,1)
%         LL.pA2(t) = LL.pA2(t) + (spm_Dpdf(A(j,:),hmm.pA(j,:,t)));
%         LL.pB2(t) = LL.pB2(t) + (spm_Dpdf(B(j,:),hmm.pB(j,:,t)));
%       %  LL.oA(t) =  LL.oA(t) + log(spm_Dpdf(B(j,:),[2 2 2]));
%     end
% end
% LL.pA2 = log(LL.pA2);


LL.pd = log(spm_Dpdf(d,hmm.pd));

LL.offline = LL.x + LL.pA(end) + LL.pB(end) + LL.pd;
LL.online = LL.xf + LL.pA(end) + LL.pB(end) + LL.pd;
LL.par = LL.pA(end) + LL.pB(end) + LL.pd;

hmm.LL = LL;

