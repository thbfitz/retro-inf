function [x,xf,xb,scale,xi] = pcog_hmm_FB(A,B,d,o,wmode);  
% function [x,xf,xb,scale,xi] = pcog_hmm_FB(A,B,d,o);  
% Calculate forward and backward sweeps (and therefore smoothing) across a
% window for HMM
%
% Inputs
% A - transition matrix
% B - observation matrix
% d - initial state vector (for windowed inference this is taken from previous filtered estimate)
% o - observations
% wmode - in windowing mode or considering whole data set (is o1 the real first observation or not?)
%
% Outputs
% x - updated (smoothed) state estimates
% xf - forward pass (filter)
% xb - backward pass
% scale - scaling factor
% xi - two-slice marginals
%
%TF 08/17

tx = size(o,2); %Length of data

% Forward inference (filtering)
%--------------------------------------------------------------------------
% xf = x;
scale=zeros(1,size(o,2));
for j=1:tx
    if j==1, 
       % xf(:,j) = d.*(B*o(:,j));
        xf(:,j) = (A'*d).*(B*o(:,j));
    else xf(:,j) = (A'*xf(:,j-1)).*(B*o(:,j));
    end
    scale(j) = sum(xf(:,j));
    xf(:,j) = xf(:,j)./scale(j);
end

% Backward inderence (beta)
%--------------------------------------------------------------------------
xb = ones(size(xf));
xb(:,tx) = xb(:,tx)*scale(tx);
for j=tx-1:-1:1 
    xb(:,j) = A*(xb(:,j+1).*(B*o(:,j+1)));  
    xb(:,j) = xb(:,j)./scale(j); %Scaling for numerical reasons
end
x= xf.*xb; %Gamma
x = x./repmat(sum(x,1),size(x,1),1);

% Dual slice marginals
%--------------------------------------------------------------------------
% xi = zeros(size(A,1),size(A,2),tx);
% for j=2:tx
%     if j==1 %For first trial
%         xi(:,:,j) = zeros(size(A,1),size(A,2));
%     else    
%         xi(:,:,j) = A.*(xf(:,j-1)*(xb(:,j).*(B*o(:,j)))'); % CHECK THIS - Possibly AN ERROR
%          A.*(xf(:,j-1)'*(xb(:,j).*(B*o(:,j)))); %
%         xi(:,:,j)=xi(:,:,j)./sum(sum(xi(:,:,j)));  
%     end
% end
xi = zeros(size(A,1),size(A,2),tx);
for j=1:tx
    if j==1 %For first trial
        if wmode % If windowing 
            xi(:,:,j) = A.*(d*(xb(:,j).*(B*o(:,j)))'); %So can learn from first time step outsid eiwndow (where this isn't t0)
            xi(:,:,j)=xi(:,:,j)./sum(sum(xi(:,:,j)));
        end
    else    
        xi(:,:,j) = A.*(xf(:,j-1)*(xb(:,j).*(B*o(:,j)))'); 
        xi(:,:,j)=xi(:,:,j)./sum(sum(xi(:,:,j)));
    end
    
end

% Alternative algoritm (Gamma recursion)
%--------------------------------------------------------------------------
% xg(:,tx) = xf(:,tx);
% for j=tx-1:-1:1
%     %jp=(xg(:,j+1)'*xf(:,j)).*A; %Calculate p(xt,xt-1|ot)
%     jp = (xf(:,j)*xg(:,j+1)').*A;
%  %   jp = jp./repmat(A*xf(tx(j),:)',1,2); 
%     jp = jp./repmat((A'*xf(:,j))',size(A,1),1); 
%     xg(:,j)=sum(jp,2); %Marginalise
% end
% x=xg;


