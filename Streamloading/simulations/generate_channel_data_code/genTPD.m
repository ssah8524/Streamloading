function y=genTPD(TPD,rType,lType)
r=rand;
ind=3*rType+lType-5;
if r==1
    j=find(TPD(ind,:)==1,1);
    y=TPD(1,j);
else
    i=find(TPD(ind,:)>r,1);
    y=(r-TPD(ind,i-1))/(TPD(ind,i)-TPD(ind,i-1))*(TPD(1,i)-TPD(1,i-1))+TPD(1,i-1); %linear interpolation 
end
return