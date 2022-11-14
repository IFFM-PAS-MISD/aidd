function Rmax=radius_max(b,Length,Width)
% radius_max - max radius for each angle in a range 0:2pi
%              in a rectangular box of dimensions Length, Width
%              with origin at the centre

lxmax=Length/2;
lymax=Width/2;
number_of_angles = length(b);
Rmax=zeros(number_of_angles ,1);
for k=1:number_of_angles 
    % first qauadrant
    if(b(k)<=pi/2)
        if(b(k) <= atan(lymax/lxmax))
            Rmax(k,1)=sqrt((lxmax*tan(b(k))).^2+lxmax^2);
        else
            Rmax(k,1)=sqrt((lymax*tan(pi/2-b(k))).^2+lymax^2);
        end
    end
    % second qauadrant
    if( b(k)>pi/2 && b(k)<=pi)
        if(b(k)-pi/2 <= atan(lxmax/lymax))
            Rmax(k,1)=sqrt((lymax*tan(b(k)-pi/2)).^2+lymax^2);
            
        else
            Rmax(k,1)=sqrt((lxmax*tan(pi-b(k))).^2+lxmax^2);
        end
    end
    % third qauadrant
    if( b(k)>pi && b(k)<=3*pi/2)
        if(b(k)-pi <= atan(lymax/lxmax))
            Rmax(k,1)=sqrt((lymax*tan(b(k)-pi)).^2+lymax^2);
            
        else
            Rmax(k,1)=sqrt((lxmax*tan(3*pi/2-b(k))).^2+lxmax^2);
        end
    end
    % fourth qauadrant
    if( b(k)>3*pi/2 && b(k)<=2*pi)
        if(b(k)-3*pi/2 <= atan(lymax/lxmax))
            Rmax(k,1)=sqrt((lymax*tan(b(k)-3*pi/2)).^2+lymax^2);
            
        else
            Rmax(k,1)=sqrt((lxmax*tan(2*pi-b(k))).^2+lxmax^2);
        end
    end
    
end