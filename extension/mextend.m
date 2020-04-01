function [ uext ] = mextend( u, mask, x, y, Nx )
%MEXTEND 

i0 = Nx/2; 
j0 = Nx/2;
dx = 1/Nx;

s_dudv=0.*x;

for s1=1:-2:-1
for s2=1:-2:-1

    for i=i0:s1: max(sign(s1)*Nx,1) 
    for j=j0:s2: max(sign(s2)*Nx,1)
        
      if mask(i,j)>0
        a=abs(i-i0);
        b=abs(j-j0);

        th=a/(a+b);

        uw=u(i-s1,j);
        us=u(i,j-s2);

%********linear extension
%         u(i,j)= s_dudv(i,j)/(a+b)+th*uw+(1-th)*us;
%******** end  linear extension

%*********** modified linear extension
           u(i,j)= double(s_dudv(i,j)>0)*s_dudv(i,j)/(a+b)+th*uw+(1-th)*us;
% % constant extension (maybe should not be better  ...
%         % u(i,j)=  th*uw+(1-th)*us;         
         u(i,j)=max(u(i,j),0.);
        %u(i,j)= th*uw+(1-th)*us;
%*********** end modified linear extension

      end
    end
    end
end
end

uext=u;