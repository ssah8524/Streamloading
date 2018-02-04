function ret=gen_quality_fn_data
clc
clear all
number_of_traces=100;%for each movie
size_of_each_trace=4500;%we need atleast these many to run simulations for 10 minutes, or else we run out
%for oceania we concatenate (the short~3000 seg) movie to obtain 4500 long trace




    for tr_ind=1:number_of_traces
        tr_ind
        
        rate=dlmread('compression_rates_from_MSSSIM_for_oceania_1sec.txt');rate=rate';
        q=dlmread('dmos_values_from_MSSSIM_for_oceania_seg_1sec.txt');q=q';
        
        rate=[rate; rate];%we concatenate the movie to obtain a longer movie
        q=[q; q];%we concatenate the movie to obtain a longer movie
        
        tmpshift=unidrnd(size(q,1));%decides the random shift for generation of data of this movie
        quality_from_quality_rate_tradeoff_data=circshift(q, tmpshift);
        quality_from_quality_rate_tradeoff_data=quality_from_quality_rate_tradeoff_data(1:size_of_each_trace,:);
        rate_from_quality_rate_tradeoff_data=circshift(rate, tmpshift);
        rate_from_quality_rate_tradeoff_data=rate_from_quality_rate_tradeoff_data(1:size_of_each_trace,:);
        
        tmp='data/Q_QR_data_N=';
        tmp=strcat(tmp,'_T=',num2str(size_of_each_trace),'_',num2str(0+tr_ind),'.txt');
        dlmwrite(tmp,quality_from_quality_rate_tradeoff_data)
        tmp='data/R_QR_data_N=';
        tmp=strcat(tmp,'_T=',num2str(size_of_each_trace),'_',num2str(0+tr_ind),'.txt');
        dlmwrite(tmp,rate_from_quality_rate_tradeoff_data)
        
        clear quality_from_quality_rate_tradeoff_data;
        clear rate_from_quality_rate_tradeoff_data;
        
        q=dlmread('PSNR_values_for_oceania_seg_1sec.txt');q=q';        
        q=[q; q];%we concatenate the movie to obtain a longer movie
        
        quality_from_quality_rate_tradeoff_data=circshift(q, tmpshift);
        quality_from_quality_rate_tradeoff_data=quality_from_quality_rate_tradeoff_data(1:size_of_each_trace,:);
        
        tmp='data/QP_QR_data_N=';
        tmp=strcat(tmp,'_T=',num2str(size_of_each_trace),'_',num2str(0+tr_ind),'.txt');
        dlmwrite(tmp,quality_from_quality_rate_tradeoff_data)
        
        clear quality_from_quality_rate_tradeoff_data;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        rate=dlmread('compression_rates_from_MSSSIM_for_route_1sec.txt');rate=rate';
        rate=[rate 900*ones(5800,1)];%we are adding redundant information for extra representation to ensure that there are 6 representations available (with one redundant representation) so that we can use the same 'runsim.m' code
        q=dlmread('dmos_values_from_MSSSIM_for_route_seg_1sec.txt');q=q';
        q=[q q(:,5).*ones(5800,1)];%we are adding redundant information for extra representation to ensure that there are 6 representations available (with one redundant representation) so that we can use the same 'runsim.m' code
        
        tmpshift=unidrnd(size(q,1));%decides the random shift for generation of data of this movie
        quality_from_quality_rate_tradeoff_data=circshift(q, tmpshift);
        quality_from_quality_rate_tradeoff_data=quality_from_quality_rate_tradeoff_data(1:size_of_each_trace,:);
        rate_from_quality_rate_tradeoff_data=circshift(rate, tmpshift);
        rate_from_quality_rate_tradeoff_data=rate_from_quality_rate_tradeoff_data(1:size_of_each_trace,:);
        
        tmp='data/Q_QR_data_N=';
        tmp=strcat(tmp,'_T=',num2str(size_of_each_trace),'_',num2str(100+tr_ind),'.txt');
        dlmwrite(tmp,quality_from_quality_rate_tradeoff_data)
        tmp='data/R_QR_data_N=';
        tmp=strcat(tmp,'_T=',num2str(size_of_each_trace),'_',num2str(100+tr_ind),'.txt');
        dlmwrite(tmp,rate_from_quality_rate_tradeoff_data)
        
        clear quality_from_quality_rate_tradeoff_data;
        clear rate_from_quality_rate_tradeoff_data;
        
        q=dlmread('PSNR_values_for_route_seg_1sec.txt');q=q';
        q=[q q(:,5).*ones(5800,1)];%we are adding redundant information for extra representation to ensure that there are 6 representations available (with one redundant representation) so that we can use the same 'runsim.m' code
   
        quality_from_quality_rate_tradeoff_data=circshift(q, tmpshift);
        quality_from_quality_rate_tradeoff_data=quality_from_quality_rate_tradeoff_data(1:size_of_each_trace,:);
        
        tmp='data/QP_QR_data_N=';
        tmp=strcat(tmp,'_T=',num2str(size_of_each_trace),'_',num2str(100+tr_ind),'.txt');
        dlmwrite(tmp,quality_from_quality_rate_tradeoff_data)
        
        clear quality_from_quality_rate_tradeoff_data;        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        rate=dlmread('compression_rates_from_MSSSIM_for_valkaama_1sec.txt');rate=rate';
        q=dlmread('dmos_values_from_MSSSIM_for_valkaama_seg_1sec.txt');q=q';
        
        tmpshift=unidrnd(size(q,1));%decides the random shift for generation of data of this movie
        quality_from_quality_rate_tradeoff_data=circshift(q, tmpshift);
        quality_from_quality_rate_tradeoff_data=quality_from_quality_rate_tradeoff_data(1:size_of_each_trace,:);
        rate_from_quality_rate_tradeoff_data=circshift(rate, tmpshift);
        rate_from_quality_rate_tradeoff_data=rate_from_quality_rate_tradeoff_data(1:size_of_each_trace,:);
        
        tmp='data/Q_QR_data_N=';
        tmp=strcat(tmp,'_T=',num2str(size_of_each_trace),'_',num2str(200+tr_ind),'.txt');
        dlmwrite(tmp,quality_from_quality_rate_tradeoff_data)
        tmp='data/R_QR_data_N=';
        tmp=strcat(tmp,'_T=',num2str(size_of_each_trace),'_',num2str(200+tr_ind),'.txt');
        dlmwrite(tmp,rate_from_quality_rate_tradeoff_data);
        
        clear quality_from_quality_rate_tradeoff_data;
        clear rate_from_quality_rate_tradeoff_data;
        
        q=dlmread('PSNR_values_for_valkaama_seg_1sec.txt');q=q';
        
        quality_from_quality_rate_tradeoff_data=circshift(q, tmpshift);
        quality_from_quality_rate_tradeoff_data=quality_from_quality_rate_tradeoff_data(1:size_of_each_trace,:);
        
        tmp='data/QP_QR_data_N=';
        tmp=strcat(tmp,'_T=',num2str(size_of_each_trace),'_',num2str(200+tr_ind),'.txt');
        dlmwrite(tmp,quality_from_quality_rate_tradeoff_data)
        
        clear rate_from_quality_rate_tradeoff_data;
    
end


'done'
