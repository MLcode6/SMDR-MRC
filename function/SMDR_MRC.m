function [W,M] = SMDR_MRC(label_data,unlabel_data,label_target,LPparam,optmParameter,r)
%% optimization parameters
lambda1          = optmParameter.lambda1;  % alpha 实例相关性
alpha            = optmParameter.alpha;    % beta 1范数
beta             = optmParameter.beta;     % gamma 2，1范数
gamma            = optmParameter.gamma;    % 标签相关性
lambda2          = optmParameter.lambda2;  % 独立性
sigma            = optmParameter.sigma;
miu              = optmParameter.miu;
maxIter          = optmParameter.maxIter;
miniLossMargin   = optmParameter.minimumLossMargin;

%% Label Propagation Parameters
lambda_l = LPparam.lambda_l;
lambda_u = LPparam.lambda_u;
K = LPparam.K;

%% LP initializtion
Y_temp = label_target';
[num_label,~] = size(label_data);
[~,c] = size(Y_temp);
[num_unlabel,~] = size(unlabel_data);
Y_unlabel = zeros(num_unlabel,c);
num_train = num_label + num_unlabel;
y_label = zeros(num_label,1);
y_unlabel = ones(num_unlabel,1);
Y_temp=[Y_temp y_label];
Y_unlabel = [Y_unlabel y_unlabel];
X = [label_data;unlabel_data];
Y = [Y_temp;Y_unlabel];

%% Graph Construction
G = pdist2(X,X);
for i =1:num_train
    temp = G(i,:);
    Gs =sort(temp);
    temp  = (temp <=  Gs(K));
    G(i,:) = temp;
end
dd = sum(G,2);
D = diag(dd);
G_ = D\G;

%% Multi-label label Propagation
Y_ = Y ./ sum(Y,2);
Y_l = Y_(1:num_label,:);
Y_u = Y_((num_label+1):num_train,:);
% G_ll = W_(1:num_label,1:num_label);
% G_lu = W_(1:num_label,(num_label+1):num_train);
G_ul = G_((num_label+1):num_train,1:num_label);
G_uu = G_((num_label+1):num_train,(num_label+1):num_train);
I = eye(num_unlabel);
I_lu = lambda_u * I;
Fl = Y_l(:,1:c);
Fu = (I - I_lu*G_uu)\(I_lu*G_ul*Y_l + (I-I_lu)*Y_u);
Fu = Fu(:,1:c);

%% label correlation
RR = pdist2(label_target+eps,label_target+eps, 'cosine' );%R 得到的只是距离，距离与相似度是成反比的
C = 1 - RR;%用1—RR得到的就是相似度矩阵
%C = abs(C);
Fu_ = Fu * C;
F_ = [Fl;Fu_];
F_ = F_';

%% 计算Z
X_temp = X';
[m,n]=size(X_temp);
[U,S,V]=svd(X_temp);
q=diag(S);
t=sum(q>0);%t=rank(Xtrn)
U1=U(:,1:t);
U2=U(:,t+1:m);
V1=V(:,1:t);
V2=V(:,t+1:n);
% miu=param.miu; %max trace(HX'PP'XHL) s.t. P'(miuXX'+(1-miu)I)P=I
Sigma=S(1:t,1:t);
A1=Sigma/(miu*(Sigma^2)+(1-miu)*eye(t,t));
A=A1^(1/2)*Sigma^(-1/2);
one=ones(n,1);
H=eye(n,n)-(1/n)*(one*one');
B=F_*H*V1*Sigma*A;
[~,~,P2]=svd(B);
Z=P2'*A*Sigma*V1';

temp_Z = Z';
cen_Z = temp_Z - repmat(mean(temp_Z,1),size(temp_Z,1),1);
if sum(sum(isnan(temp_Z)))>0
    temp_Z = Z'+eps;
    cen_Z = temp_Z - repmat(mean(temp_Z,1),size(temp_Z,1),1);
end

%% optimization initializtion
num_dim = size(X,2);
% XT = X';
XTX = X'*X;
XTZ = X'*cen_Z;
% 1范数约束的W
W   = (XTX + sigma*eye(num_dim)) \ (XTZ);
W_1 = W;
% 2，1范数约束的M
M   = (XTX + sigma*eye(num_dim)) \ (XTZ);
M_1 = M;

%% label correlation
function L = label_correlation(Y)
    R     = pdist2( Y'+eps, Y'+eps, 'cosine' );
    C = 1 - R;
    C = abs(C);
    L = diag(sum(C,2)) - C;
end
%% instance correlation
% C = ins_similarity(X,10);
L1 = diag(sum(G,2)) - G;
%% Iterative  
iter = 1; 
oldloss = 0;
tk = 1;
tk_1 = 1;

% HSIC用到的参数
n1=size(W,1);
I1=eye(n1,n1);

Wh = W';
[~,n2] = size(Wh);
one=ones(n2,1);
% H1 = eye(n2,n2)-(1/n2)*(one*one');

% 标签相关性
L2 = label_correlation(cen_Z);

%% 计算LIP
A = gradL21(M);
varepsilon = 0.01;
Lf = sqrt(8*(norm(XTX)^2 + 6*gamma*norm(L2)^2)+ 4*beta*norm(A)^2);
s1 = varepsilon*sqrt(2*alpha);
s2 = num_dim*sqrt(alpha/2)+sqrt((num_dim^2*alpha/2)+Lf*varepsilon);
mu=s1/s2;
    
Lip=Lf+(alpha*num_dim)/mu;

%% s-proximal gradient(S-APG)
while iter <= maxIter
    A = gradL21(M);
    XTX = X'*X;
    XTZ = X'*cen_Z;
    
    Zeta_Wk  = W + (tk_1 - 1)/tk * (W - W_1);
    Zeta_Mk  = M + (tk_1 - 1)/tk * (M - M_1);
    
    % calculate the graid of F_mu 
    grad_M_F=XTX*Zeta_Wk+XTX*Zeta_Mk-XTZ+beta*A*Zeta_Mk+gamma*Zeta_Mk*L2;
       
    grad_W_F_1=XTX*Zeta_Wk+XTX*Zeta_Mk-XTZ+gamma*Zeta_Wk*L2;
    PS=softthres(alpha*Zeta_Wk,mu);
    grad_W_F_2=(alpha^2/mu)*Zeta_Wk-(alpha/mu)*PS;
    grad_W_F=grad_W_F_1+grad_W_F_2;
             
    % calculate W(k),M(k)
    r1=(1/Lip);
    Wk=Zeta_Wk-r1*grad_W_F;
    Mk=Zeta_Mk-r1*grad_M_F;
    
    % 更新 tk，W, M
    tk_1=tk;
    tk=(1 + sqrt(4*tk^2 + 1))/2;
       
    W_1=W;
    M_1=M;
    
    % calculate W^k,M^k
    q1=lambda2/Lip;
    q2=lambda1/Lip;
       
    W = (q1*(M*M')+q2*X'*L1*X+I1)\(Wk-q2*X'*L1*X*M);
    M = (q1*(W*W')+q2*X'*L1*X+I1)\(Mk-q2*X'*L1*X*W);

    %% 计算损失函数的值
    O1 = (X*(M+W) - cen_Z);
    DiscriminantLoss = (1/2)*trace(O1'* O1);
    WM_correlationloss = (lambda2/2)*trace((W'*M)' * (W'*M));
    L_correlationloss = (gamma/2)*(trace(M*L2*M')+trace(W*L2*W'));
    sparsity1    = alpha*norm(W,1);
    sparsity2    = beta*trace(M'*A*M);
    sample_correlationloss = (lambda1/2)*trace((X*(W+M))'*L1*(X*(W+M)));

    totalloss = DiscriminantLoss + WM_correlationloss + sparsity1 + sparsity2 + L_correlationloss + sample_correlationloss;
       
    loss(iter,1) = totalloss;
    if abs(oldloss - totalloss) <= miniLossMargin
        %本次迭代的结果与上次的结果相差少于预订的最小损失间距时，结束循环
        break;
    elseif totalloss <=0
        break;
    else
        oldloss = totalloss;
    end
    
    iter=iter+1;
end
W = W(:,1:r);
M = M(:,1:r);
end

%% soft thresholding operator
function W = softthres(W_t,lambda)
    W = max(W_t-lambda,0) - max(-W_t-lambda,0);  
end

function A = gradL21(P)
num = size(P,1);
A = zeros(num,num);
for i=1:num
    temp = norm(P(i,:),2);
    if temp~=0
        A(i,i) = 1/temp;
    else
        A(i,i) = 0;
    end
end
end
