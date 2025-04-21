function [r,t,X]=deserialize(x, n)
    rt=reshape(x(1:6*n),6,[]);
    r=reshape(rt(1:3,:),3,[]);
    t=reshape(rt(4:6,:),3,[]);
    x=x(6*n+1:end);
    X=reshape(x,3,[]);
end