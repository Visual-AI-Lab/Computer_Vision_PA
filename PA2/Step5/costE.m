function [F]=costE(x, param)
    nimg = length(param.uv); % Number of camera poses.
    uv = param.uv;
    K = param.K;  
    
    % Extract R, T, X
    [Rvec,Tvec,X] = deserialize(x,nimg);
    nXn=0;
    for i=1:nimg
        nXn = nXn + length(uv{i}); end %number of reprojection errors
    
    F = zeros(2*nXn,1); 
    
    count = 1;
    for i = 1:nimg        
        % Rotation, Translation, [X, Y, Z]
        X_idx = uv{i}(3,:); nXi = size(X_idx, 2);
        R = RotationVector_to_RotationMatrix(Rvec(:,i)); T = Tvec(:,i); Xi = X(:,X_idx);   

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % write code to calculate reprojection errors and store them into
        % variable F
        
        for j = 1:nXi
            X_proj = K * (R * Xi(:,j) + T);
            X_proj = X_proj / X_proj(3);
            F(count) = X_proj(1)-uv{i}(1,j);
            F(count+1) = X_proj(2)-uv{i}(2,j);
            count = count + 2;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end