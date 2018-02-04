%for num_users=[15 18 21 24 27 30 33] gen_data(num_users); end


function ret=gen_channel_data_mine(num_users)
total_number_slots=150000;
'generating channel for'
num_users
homogeneous_channels=1;
markov_channels=1;  mkv_channel_granularity=10;
ch_scale=1;
load TPD; %matrix with data for channel gain random variable

for r = 1:20
    ch_gain=zeros(num_users,total_number_slots);
    % get channel pdf
    ind=3;
    indstart=find(TPD(ind,:)~=0,1)-1;
    indend=find(TPD(ind,:)==1,1);
    ChannelPDF=zeros(2,mkv_channel_granularity*(indend - indstart));
    for i=indstart:indend-1
        for j=1:mkv_channel_granularity
            %Assuming a uniform distribution between different amounts of
            %throughput for different granularity levels.
            ChannelPDF(:,(i-indstart)*mkv_channel_granularity + j)=...
                [(TPD(1,i+1)-TPD(1,i))/mkv_channel_granularity*j+TPD(1,i);(TPD(ind,i+1)-TPD(ind,i))/mkv_channel_granularity];
        end
    end
    
    for t=1:total_number_slots
        if mod(t,5000)==0
            t
        end
        if markov_channels==0
            for i=1:num_users
                ch_gain(i,t)=ch_scale*genTPD(TPD,2,2);%using Zheng's function to generate channel gain random variable
            end
        else
            if t==1
                for i=1:num_users
                    ch_gain(i,t)=genTPD(TPD,2,2);%using Zheng's function to generate channel gain random variable
                    ch_gain(i,t)=round(ch_gain(i,t)/(250*ch_scale/mkv_channel_granularity))*(250*ch_scale/mkv_channel_granularity); % ensure that the granularity is taken care of
                end
            else
                for i=1:num_users
                    %do appropriate scaling to undo all the scaling that is
                    %caused by the heterogenienty, and channel gain scaling
                    c_new = mcchannel_mine(  round((ch_gain(i,t-1)/ch_scale)*...
                        ( 1- min(1/3,(homogeneous_channels==0)* (ceil(3*i/num_users) -2)))),ChannelPDF,10);%using Zheng's function to generate channel gain random variable for markov channels
                    ch_gain(i,t)=ch_scale*c_new;
                end
            end
        end
        
        if homogeneous_channels==0 %some channels are worse and some are better than average
            ch_gain(:,t)=ch_gain(:,t).*[[.5*ones(1,num_users/3)] [ones(1,num_users/3)] [1.5*ones(1,num_users/3)] ]' ;
        end
        
    end
    tmp='data/iid/chdata_N=';
    tmp=strcat(tmp,num2str(num_users),'_T=',num2str(total_number_slots),'_round=',num2str(r));
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
    dlmwrite(tmp,ch_gain)
    'done'
end
end