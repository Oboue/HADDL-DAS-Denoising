% function [d] = reconstruction(din,mask,dt,fmin,fmax,rank_p,alpha,n_iter)
% 
% aux = din;
% 
% for k = 1:n_iter;
%     
%     out = fx_ssa(aux,dt,rank_p,fmin,fmax);
%     
%     d = alpha(n_iter)*din + (1-alpha(n_iter)*mask).*out + (1-mask).*out;
%     
%     aux = d;
% end;

function [d] = reconstruction(din,T,dt,fmin,fmax,rank_p,alpha,n_iter);

aux = din;

for k = 1:n_iter;
    
    out = fx_ssa(aux,dt,rank_p,fmin,fmax);
    
    d = alpha*din + (1-alpha*T).*out;
    
    aux = d;
end;
