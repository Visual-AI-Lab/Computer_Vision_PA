function [x_BA] = LM2_iter_dof(x, param)
    x = transpose(x);
    iter = 0;
    tolX = 5e-3;   % x의 종료허용오차
    tolFun = 5e-3; % 함수값의 종료허용오차
    tolJ = 1e-5;   % Jacobian의 종료허용오차
    max_iter = 50;    
    x_BA = cell(0,1);
    
    nimgs = length(param.uv); nX = param.nX; %length(param.X_img); % nimgs=카메라 수, nX=point수
    key1 = param.key1; key2 = param.key2; % 고정 파라미터
    opt = param.optimization; dof = param.dof_remove;
    dof_idx = [ 6*key1-5 : 6*key1];
    %dof_idx = [ 6*key1-5 : 6*key1, 6*key2-2];
    while iter < max_iter
        iter = iter+1;        
        [F,J] = JacobiancostE(x, param); 
        if dof == 1, J(:,dof_idx) = []; end
        
        before_cost = norm(F)^2;
        
        H = J'*J;
        JtF = J'*F;
        
        if opt==1      % Levenberg - Marquarbt
            if iter==1, lambda = 1e-3; end
            H_LM = H + lambda*sparse(diag(diag(H)));            
        elseif opt==0  % Levenberg
            if iter==1, lambda = 1e-3*mean(diag(H)); end
            H_LM = H + lambda*eye(size(H,1));                   
        end     
        dp = -H_LM \ JtF;
        if isnan(sum(dp)) % NaN이면 그냥 끝내자
            break;
        end
        if dof == 1
            dp_idx = 1:6*nimgs+nX; dp_idx(dof_idx) = [];
            dp_temp = zeros(6*nimgs+nX,1); dp_temp(dp_idx) = dp; 
            x_LM = x+dp_temp;
        elseif dof == 0
            x_LM = x+dp;
        end

        disp(['size(x_LM) : ', mat2str(size(x_LM))]);
        after_F=costE(x_LM, param);
        after_cost = norm(after_F)^2;
        l2_p  = norm(x);
        l2_dp = norm(dp); %dp의 변화량    
            
        if (after_cost < before_cost)
            lambda=lambda*0.1; disp(['lambda : ', num2str(lambda)]);
            x=x_LM;
        else
            while after_cost >= before_cost
                disp(['after_cost : ', mat2str(after_cost)]);
                disp(['before_cost : ', mat2str(before_cost)]);
                lambda=lambda*10; disp(['lambda : ', num2str(lambda)]);
                
                if opt==1,     H_LM = H + lambda*sparse(diag(diag(H))); 
                elseif opt==0, H_LM = H + lambda*eye(size(H,1));    
                end        
                
                dp = -H_LM \ JtF;
                
                if dof == 1
                    dp_idx = 1:6*nimgs+nX; dp_idx(dof_idx) = [];
                    dp_temp = zeros(6*nimgs+nX,1);  dp_temp(dp_idx) = dp;
                    x_LM = x+dp_temp;
                elseif dof == 0
                    x_LM = x+dp;
                end
                
                after_F=costE(x_LM, param);
                after_cost = norm(after_F)^2;
            end
            x = x_LM;
        end        
        disp(['Iteration: ',num2str(iter),'   Cost value is ',num2str(before_cost)]);
        
        %if ( abs(before_cost - after_cost) <= tolFun )        
        if ( abs(before_cost - after_cost) < tolFun*before_cost )
            disp('Finished (tolFun)');
            x_BA{end+1} = x_LM;
            break;
        end
        
        if ( l2_dp < tolX*l2_p )
            disp('Finished (tolX)');
            x_BA{end+1} = x_LM;
            break;
        end
                
        if mod(iter,5)==0 || iter<10
            x_BA{end+1} = x_LM;
        end
    end

    if iter==max_iter
        x_BA{end+1} = x_LM;
        disp('Finished (max_iter)');
    end

end