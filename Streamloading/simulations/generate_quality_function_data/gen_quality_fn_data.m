%Amir: The purpose of this function is to simply take the data from the
%measurement files and for each of the users in the system, create a random
%shift of the segments to be downloaded, and store the rate and quality in
%order for the main program tu run.

function ret=gen_quality_fn_data(num_users)

total_number_segments=4500;

rate=dlmread('compression_rates_from_MSSSIM_for_valkaama_1sec.txt');rate=rate';
q=dlmread('dmos_values_from_MSSSIM_for_valkaama_seg_1sec.txt');q=q';

%Amir: the below matrices are 3 dimensional matrices with entries having
%the quality/rate of each user for each segment.
quality_from_quality_rate_tradeoff_data=zeros(num_users,total_number_segments,size(q,2));
rate_from_quality_rate_tradeoff_data=zeros(num_users,total_number_segments,size(q,2));

for user_index=1:num_users
    tmp=unidrnd(2*num_users);%decides the random shift for generation of data of this movie
    quality_from_quality_rate_tradeoff_data(user_index,:,:)=circshift(q, tmp*( floor(size(q,1)/(2*num_users))-1));    
    rate_from_quality_rate_tradeoff_data(user_index,:,:)=circshift(rate,tmp*( floor(size(rate,1)/(2*num_users))-1));    
end

tmp='data/quality_QR_data_N=';
tmp=strcat(tmp,num2str(num_users),'_T=',num2str(total_number_segments));
dlmwrite(tmp,quality_from_quality_rate_tradeoff_data)
tmp='data/rate_QR_data_N=';
tmp=strcat(tmp,num2str(num_users),'_T=',num2str(total_number_segments));
dlmwrite(tmp,rate_from_quality_rate_tradeoff_data)
'done'
