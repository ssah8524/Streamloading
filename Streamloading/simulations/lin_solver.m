user_no = 3;
layer_no = 5;
seg_no = 10;

rmin = 0.001;
svc_overhead = 0.1;
t_slot = 0.01;
t_seg = 1;
time_no = seg_no/t_slot;

tmp='generate_quality_function_data/data/quality_QR_data_N=';
tmp=strcat(tmp,num2str(33),'_T=',num2str(4500));
quality_QR_data=dlmread(tmp);
quality_QR_data=100-quality_QR_data;
quality_QR_data=reshape(quality_QR_data,33,4500,6);
quality_QR_data=quality_QR_data(   [1:(0+(user_no/3)) 12:(11+(user_no/3)) 23:(22+(user_no/3))]          ,:,:);
quality_QR_data = quality_QR_data(1:user_no,1:seg_no,:);

tmp='generate_quality_function_data/data/rate_QR_data_N=';
tmp=strcat(tmp,num2str(33),'_T=',num2str(4500));
rate_QR_data=dlmread(tmp);
rate_QR_data=reshape(rate_QR_data,33,4500,6);
rate_QR_data=(1+svc_overhead)*rate_QR_data(   [1:(0+(user_no/3)) 12:(11+(user_no/3)) 23:(22+(user_no/3))]          ,:,:);
rate_QR_data = rate_QR_data(1:user_no,1:seg_no,:);

layerwise_size=zeros(user_no,seg_no,layer_no);
layerwise_q=zeros(user_no,seg_no,layer_no);

for l=1:5
    layerwise_q(:,:,l) = quality_QR_data(:,:,(l+1)) - quality_QR_data(:,:,l);
    layerwise_size(:,:,l) = rate_QR_data(:,:,(l+1)) - rate_QR_data(:,:,l);
end
qual_per_size = layerwise_q./layerwise_size;

homogeneous_channels=1;
markov_channels=1;
tmp='generate_channel_data_code/data/chdata_N=';
tmp=strcat(tmp,num2str(33),'_T=',num2str(150000));
if homogeneous_channels==1
    tmp=strcat(tmp,'_homo');
else
    tmp=strcat(tmp,'_het');
end
if markov_channels==0
    tmp=strcat(tmp,'_iid');
else
    tmp=strcat(tmp,'_mkv');
end
ch_gain_scale=1.5;
ch_gain=ch_gain_scale*t_slot*dlmread(tmp)+30*rmin;
tmp=size(ch_gain,1)*size(ch_gain,2);

ch_gain=(reshape(ch_gain,tmp/33,33))';
%ch_gain=(reshape(ch_gain,33,tmp/33));
ch_gain=ch_gain(    [1:(0+(user_no/3)) 12:(11+(user_no/3)) 23:(22+(user_no/3))]        , : ) ;
ch_gain = ch_gain - 100*(1+svc_overhead)*t_slot; %%Currently only works with the smallest base layer size!!
ch_gain = ch_gain((1:user_no),(1:time_no));


x = zeros(time_no*user_no*seg_no*layer_no,1);
A = zeros(time_no + seg_no*user_no,length(x));
b = zeros(time_no + seg_no*user_no,1);
Aeq = zeros(length(x));
beq = zeros(length(x),1);
f = zeros(length(x),1);

tic
for k = 1:time_no
   A(k,((k-1)*seg_no*user_no*layer_no + 1):(k*seg_no*user_no*layer_no)) = 1; 
   b(k) = 1;
end

for i = 1:user_no
    for j = 1:seg_no
        for t = 1:time_no
            for r = 1:layer_no
                A(time_no + (i-1)*seg_no + j,r + (t-1)*(layer_no*seg_no*user_no) + (i-1)*(layer_no*seg_no) + ...
                    (j-1)*layer_no) = t_slot * ch_gain(i,t);
                if t_slot*t > t_seg*j
                    Aeq(r + (t-1)*(layer_no*seg_no*user_no) + (i-1)*(layer_no*seg_no) + ...
                    (j-1)*layer_no,r + (t-1)*(layer_no*seg_no*user_no) + (i-1)*(layer_no*seg_no) + ...
                    (j-1)*layer_no) = 1;
                end
                f(r + (t-1)*(layer_no*seg_no*user_no) + (i-1)*(layer_no*seg_no) + ...
                    (j-1)*layer_no) = t_slot * ch_gain(i,t) * qual_per_size(i,j,r);
            end
        end
        b(time_no + (i-1)*seg_no + j) = sum(layerwise_size(i,j,:));
    end
end
toc

tic
[X,FVAL] = linprog(-f,A,b,Aeq,beq,zeros(length(x),1),Inf);
toc

-FVAL




