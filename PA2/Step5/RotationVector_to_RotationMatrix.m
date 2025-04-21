function R = RotationVector_to_RotationMatrix(r)

theta = norm(r,2);
if theta == 0
    n = [0;0;0];
else
    n = r/theta;
end
n_cross = [0, -n(3), n(2);
          n(3), 0, -n(1);
          -n(2), n(1), 0]; 

R = eye(3) + sin(theta)*n_cross + (1-cos(theta))*n_cross*n_cross;

end