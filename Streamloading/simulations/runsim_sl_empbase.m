%important_constant_1: importance of variability;
%discretization will cause following problems:
%if too large will not allow increase in quality even when no rebuffering issues

%important_constant_2
%decides the sensitivity of corrections in b_est

function return_status_INDEX=runsim_sl_empbase(fnarg)

%Amir: determining the global variables. For now, we will be using the
%simple linear function instead of the introduced lipschitz function.
global base_size_index num_users eta_RVBAR rmin_RVBAR m_est v_est b_base_est b_enh_est d_est rho_est sigma_est epsilon  user_present...
    count_rebuff_slots beta_base_bar beta_enh_bar slot total_enh t_seg t_slot size_of_current_segment_being_downloaded...
    total_base svc_overhead b_base_init b_enh_init fn_b_scale

%Amir: Take the input vector fnarg and assign values for respective variables.
%close all;clc;
min_seg_sim=600;num_users=fnarg(1);
rate_allocation_scheme=fnarg(2);quality_adaptation_scheme=fnarg(3);
homogeneous_channels=fnarg(4);markov_channels=fnarg(5);
pbar_val=fnarg(6);max_base_segments=fnarg(8);


t_seg=1;%in seconds
t_slot=0.01;%in seconds
svc_overhead = 0.1;

%initial_num_segments_to_be_downloaded_at_low_quality=0; %%Amir: What is this for? (In order to fill up the buffer to some extent?)
%Amir: the below line uses an initial proportional fairness before starting
%to run the actual algortihm.
LPF_run_slots_before_using_RVBAR=0;%needed to ensure that there is no rebuffering in the beginning for the weakest users due to poor allocation
segments_to_run_greedy_before_QVBAR=0;%Amir: what for ?
variability_importance_constant=.1;%important_constant_1. Amir: is that the same eta?

count_rebuff_slots=zeros(1,num_users);
user_present=ones(1,num_users);%Amir: what does this represent?


%Amir: For streamloading, the betha parameter is only used for the baselayer resource allocation.
beta_base_bar=(0)*ones(1,num_users);%-(1/21)*ones(1,num_users);
%beta_base_bar=min(-1/21+max(slot-(10^4),0)*(10^-4)/51,-1/51)*ones(1,num_users);
beta_enh_bar=(0)*ones(1,num_users);

%Amir: The base_size_index determines the size of the base layer with
%respect to the whole segment size. In this regard, it indexes the
%respective encoding size among the 6 available compression rates.
base_size_index = 1;
%base_size_index=ones(1,num_users);


b_base_init=15*ones(1,num_users);
b_enh_init=15*ones(1,num_users);
fn_b_scale=0.01*ones(1,num_users);

%Amir: This is an indicator of the number of outlying base layer segments
%for each user which is incremented periodically and decremented after a
%base layer segment download completion. The assumption is that the
%download begins with a full base layer buffer, where the last allowable base layer segment finished
%downloading at t=0. Hence, at t=0, the next base layer segment becomes
%available.
base_queue=5*ones(1,num_users);

d_init=1*ones(1,num_users);
total_allocated_rate=0*ones(1,num_users);

%Amir: I suppose the 2 below lines are just an initialization.
size_of_current_segment_being_downloaded=100*ones(1,num_users);
quality_of_current_segment_being_downloaded=20*ones(1,num_users);

epsilon=0.05;
%Amir: then, what is the variability importance?
eta_RVBAR =.01;
%Amir: Is it in units of Mb per slot?
rmin_RVBAR=.001;

%Amir: Let's try it first without data cost.
xi=.01;%Amir: Unit cost for data transmission (p_i^d)
%pbar=pbar_val*ones(1,num_users);
%delta_square_root_approx=0.01;

%parameter_initialization
m_est=20*ones(1,num_users);
v_est=5*ones(1,num_users);
b_base_est=b_base_init;
b_enh_est=b_enh_init;
d_est=d_init;

%Amir: what is sgima?
sigma_est=100*ones(1,num_users);
%Amir: Rho is the average throughput.
rho_est=100*ones(1,num_users);


tmp='generate_quality_function_data/data/quality_QR_data_N=';
tmp=strcat(tmp,num2str(33),'_T=',num2str(4500));
quality_QR_data=dlmread(tmp);
%Amir: What is the purpose of the following three lines?


quality_QR_data=100-quality_QR_data;
quality_QR_data=reshape(quality_QR_data,33,4500,6);
quality_QR_data=quality_QR_data(   [1:(0+(num_users/3)) 12:(11+(num_users/3)) 23:(22+(num_users/3))]          ,:,:);

tmp='generate_quality_function_data/data/rate_QR_data_N=';
tmp=strcat(tmp,num2str(33),'_T=',num2str(4500));
rate_QR_data=dlmread(tmp);
rate_QR_data=reshape(rate_QR_data,33,4500,6);
rate_QR_data=rate_QR_data(   [1:(0+(num_users/3)) 12:(11+(num_users/3)) 23:(22+(num_users/3))]          ,:,:);

tmp='generate_channel_data_code/data/chdata_N=';
tmp=strcat(tmp,num2str(33),'_T=',num2str(175000));
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
ch_gain=ch_gain_scale*t_slot*dlmread(tmp)+30*rmin_RVBAR;
tmp=size(ch_gain,1)*size(ch_gain,2);
ch_gain=(reshape(ch_gain,tmp/33,33))';

ch_gain=ch_gain(    [1:(0+(num_users/3)) 12:(11+(num_users/3)) 23:(22+(num_users/3))]        , : ) ;

traces_needed=0;
if traces_needed==1
    store_m_est_data=zeros(6,size(ch_gain,2));
    store_b_est_data=zeros(6,size(ch_gain,2));
    store_d_est_data=zeros(6,size(ch_gain,2));
    store_rho_est_data=zeros(6,size(ch_gain,2));
    %store_quality_of_current_segment_being_downloaded=zeros(6,size(ch_gain,2));
    store_quality_of_current_segment_being_downloaded=zeros(6,size(ch_gain,2));
    max_quality_current_segment_being_downloaded=zeros(6,size(ch_gain,2));
    i5_quality_current_segment_being_downloaded=zeros(6,size(ch_gain,2));
    i4_quality_current_segment_being_downloaded=zeros(6,size(ch_gain,2));
    i3_quality_current_segment_being_downloaded=zeros(6,size(ch_gain,2));
    i2_quality_current_segment_being_downloaded=zeros(6,size(ch_gain,2));
    min_quality_current_segment_being_downloaded=zeros(6,size(ch_gain,2));
end

quality_index=zeros(min_seg_sim+1,num_users);
tot_qual_matrix=zeros(3*(min_seg_sim+1),num_users);
seg_size=zeros(min_seg_sim+1,num_users);

%trace_info = zeros(14,10000*30);
%intitalization
slot=0;%index for the current slot
total_enh=-1*ones(1,num_users);
total_base=zeros(1,num_users);
base_size_data_matrix = rate_QR_data(:,:,base_size_index);
pending_enh=seg_size(1,:);
pending_base=(1 + svc_overhead)*base_size_data_matrix(:,max_base_segments + 1)';

time_counter_base = 0; %keep track of how many enhancement layers are missed because of base layer download.
out_of_time_enh = zeros(1,num_users);
qual_sel = zeros(1,num_users);
qual_sel_q = zeros(1,num_users);
rm_counter = zeros(1,num_users);

while min(total_base)<=min_seg_sim
        
    slot=slot+1;
    rm_counter(rm_counter > 0) = rm_counter(rm_counter > 0) - 1;
    %Amir: periodically, new base layer segment become available to
    %download at the base station.
    if mod(slot,t_seg/t_slot)==0
        full_base = find(pending_base <= 0 & base_queue == 0);
        if(~isempty(full_base))
            pending_base(full_base) = pending_base(full_base) + ...
                (1+svc_overhead)*diag(base_size_data_matrix(full_base,...
                total_base(full_base),base_size_index))';
        end
        base_queue = base_queue + 1;

        %Amir: Here there must be some expression to take of the case in
        %which base_queue exceeds the buffer limit in which case the base
        %layer buffer becomes empty.
        
        if(max(base_queue) >= max_base_segments)
            'Base Layer is starving';
        end
    end
    
    %Amir: flag is raised if there are base layers to download for any
    %user
    base_flag=0;
    if (sum(base_queue)~=0)
        time_counter_base = time_counter_base + 1;
        base_flag=1;
    end
    
    %rate allocation code goes here
    if rate_allocation_scheme==-1
        rate_alloc=10*rand(1,num_users);%scheme for testing code
    elseif rate_allocation_scheme==0%RVBAR
        if slot<LPF_run_slots_before_using_RVBAR %Amir:Use proportional fairness.
            'PF';
            rate_alloc=PF_rate_allocation(ch_gain(:,slot),base_queue);
            
        else
            'RVBAR';
            rate_alloc=RVBAR(ch_gain(:,slot),base_queue);
        end
        epsilon_slot=max(.001,1/slot);
    elseif rate_allocation_scheme==1%PF scheme (slot level fairness)
        rate_alloc=PF_rate_allocation(ch_gain(:,slot),base_queue);
        epsilon_slot=.1;
    end
    total_allocated_rate=total_allocated_rate+rate_alloc;
    rho_est=rho_est+epsilon_slot*( (rate_alloc/t_slot) - rho_est);
    b_base_est=b_base_est+t_slot./(t_seg.*(1+beta_base_bar));
    b_enh_est=b_enh_est+t_slot./(t_seg.*(1+beta_enh_bar));
    
    if base_flag
        
        pending_base=pending_base - rate_alloc;
        finish_base_cur=find(pending_base <= 0 & base_queue ~= 0);
        
        total_base(finish_base_cur) = total_base(finish_base_cur) + 1;
        base_queue(finish_base_cur) = base_queue(finish_base_cur) - 1;
        tot_qual_matrix(total_base(finish_base_cur),finish_base_cur) = ...
                    max(quality_QR_data(finish_base_cur,total_base(finish_base_cur),base_size_index),...
                    tot_qual_matrix(total_base(finish_base_cur),finish_base_cur));

        b_base_est(finish_base_cur) = max(b_base_est(finish_base_cur)-1,0);
        finish_base_res = find(pending_base <= 0 & base_queue ~= 0);
        
        if(~isempty(finish_base_res))
            pending_base(finish_base_res) = pending_base(finish_base_res)+...
                (1+svc_overhead)*diag(base_size_data_matrix(finish_base_res,...
                total_base(finish_base_res),base_size_index))';
        end
        
    else
        %This section is for the case when the recent base layer download
        %has taken too long.
        if (time_counter_base > t_seg/t_slot)
            front_enh = find(total_base - total_enh > 0);
            if ~isempty(front_enh)
                for ind = front_enh
                    tot_qual_matrix((max(1,total_enh(ind)+1)):total_base(ind),ind)=...
                        quality_QR_data(ind,(max(1,total_enh(ind)+1)):total_base(ind),base_size_index);
                    total_enh(ind) = total_base(ind)-1;
                    b_enh_est(ind) = max(b_enh_est(ind) - (total_base(ind)-total_enh(ind)),0);
                end
            end
            pending_enh = 0 - rate_alloc;
        else
            pending_enh = pending_enh-rate_alloc;
        end
        finish_enh = find(pending_enh <= 0);
        time_counter_base = 0;
        %Amir: This whole loop is only for those users who have
        %completely downloaded their latest segment.
        for user_index = finish_enh
            
            %Out of time delivery of enhancement layers
            if total_enh(user_index)*t_seg/t_slot < slot && slot > 100
                out_of_time_enh(user_index) = out_of_time_enh(user_index) + 1;
                tot_qual_matrix((total_enh(user_index)+1):(total_enh(user_index)+ceil(slot * t_slot/t_seg)),user_index) = ...
                    quality_QR_data(user_index,(total_enh(user_index)+1):(total_enh(user_index)+ceil(slot * t_slot/t_seg)),base_size_index);
                total_enh(user_index) = ceil(slot * t_slot/t_seg)-1;
            end
            
            flag = false;
            if quality_adaptation_scheme==0 %QVBAR                
                while (flag == false)
                    total_enh(user_index)=total_enh(user_index)+1;
                    qual_val=reshape(quality_QR_data(user_index,total_enh(user_index)+1,:),1,6);
                    last_sigma=(1+svc_overhead)*reshape(rate_QR_data(user_index,total_enh(user_index)+1,:),1,6);
                    sigma_enh=last_sigma-last_sigma(base_size_index);
                    if total_enh(user_index)<segments_to_run_greedy_before_QVBAR
                        selected_representation_index=max(1,sum((sigma_enh<=((1+beta_enh_bar(user_index))*rho_est(user_index)))));
                        user_quality=qual_val(selected_representation_index);
                        enh_size_for_quality_choice=sigma_enh(selected_representation_index);
                    else
                        
                        var_imp_scaling=variability_importance_constant*min(1,(total_enh(user_index))^2/60^2);
                        %Enable the following lines in case we want to use the
                        %lipschitz values for vurtual buffers.
                        fn_b_enh=(1/2)*max(b_enh_est(user_index),0)+(1/2)*...
                            max(b_enh_est(user_index)-b_enh_init(user_index)/2,0)...
                            +100*max(b_enh_est(user_index)-b_enh_init(user_index),0);
                        fn_b_enh=fn_b_enh*fn_b_scale(user_index);
                        tmp_metric=( qual_val) - (fn_b_enh/((1+beta_enh_bar(user_index))*t_seg)).*sigma_enh...
                            - var_imp_scaling *( ( qual_val - m_est(user_index)).^2 );
                        
                        [~, selected_representation_index]=max(tmp_metric);
                        
                        user_quality=qual_val(selected_representation_index);
                        enh_size_for_quality_choice=sigma_enh(selected_representation_index);
                        qual_sel(user_index) = selected_representation_index;
                        qual_sel_q(user_index) = user_quality;
                        
                        if selected_representation_index ~= base_size_index
                            flag = true;
                        else
                            b_enh_est(user_index)=max(b_enh_est(user_index)-1,0);
                        end
                        quality_index(total_enh(user_index)+1,user_index)=...
                            selected_representation_index;
                        tot_qual_matrix(total_enh(user_index)+1,user_index) = user_quality;
                        seg_size(total_enh(user_index)+1,user_index)=...
                            enh_size_for_quality_choice + 110;
                        size_of_current_segment_being_downloaded(user_index)=enh_size_for_quality_choice;
                        quality_of_current_segment_being_downloaded(user_index)=user_quality;
                        pending_enh(user_index) = pending_enh(user_index)+enh_size_for_quality_choice;
                    end
                end
            end
            if quality_adaptation_scheme==1 && rm_counter(user_index) <= 0%greedy segment selection based on current estimate of throughput
                total_enh(user_index)=total_enh(user_index)+1;
                qual_val=reshape(quality_QR_data(user_index,total_enh(user_index)+1,:),1,6);
                last_sigma=(1+svc_overhead)*reshape(rate_QR_data(user_index,total_enh(user_index)+1,:),1,6);
                sigma_enh=last_sigma-last_sigma(base_size_index);
                
                selected_representation_index=max(1,sum((sigma_enh<=((1-.01)*rho_est(user_index)))));
                user_quality=qual_val(selected_representation_index);
                enh_size_for_quality_choice=sigma_enh(selected_representation_index);
                if selected_representation_index == 1
                    rm_counter(user_index) = 10;
                end
                quality_index(total_enh(user_index)+1,user_index)=...
                    selected_representation_index;
                tot_qual_matrix(total_enh(user_index)+1,user_index) = user_quality;
                seg_size(total_enh(user_index)+1,user_index)=...
                    enh_size_for_quality_choice + 110;
                size_of_current_segment_being_downloaded(user_index)=enh_size_for_quality_choice;
                quality_of_current_segment_being_downloaded(user_index)=user_quality;
                pending_enh(user_index) = pending_enh(user_index)+enh_size_for_quality_choice;
            end
            %Amir: The rest of the parameters are updated here in case of
            %segment completetion.
            epsilon_seg=epsilon;%max(epsilon,1/(total_number_of_segments_downloaded_by_users(user_index)));
            m_est(user_index)=m_est(user_index) + epsilon_seg*( user_quality- m_est(user_index) );
            v_est(user_index)=v_est(user_index) + epsilon_seg*( (user_quality - m_est(user_index))^2 - v_est(user_index)  );
            sigma_est(user_index) = sigma_est(user_index) + epsilon_seg*(enh_size_for_quality_choice -  sigma_est(user_index));
            
            %Amir: How should the virtual buffer be updated.
            %b_est(user_index)=b_est(user_index)+.0001*(  (sigma_est(user_index)./((1+beta_bar(user_index))*t_seg))  - rho_est(user_index));
            %b_est(user_index)=b_est(user_index)+.01*sign(  (sigma_est(user_index)./((1+beta_bar(user_index))*t_seg))  - rho_est(user_index));
            b_enh_est(user_index)=max(b_enh_est(user_index)-1,0);
        end
    end
    count_rebuff_slots=count_rebuff_slots+ (( ( (slot-count_rebuff_slots)*t_slot/t_seg)- total_base)-3>0);
    if traces_needed==1
        store_rho_est_data(:,slot)=[user_present(1:2) user_present(6:7) user_present(11:12)].*[rho_est(1:2) rho_est(6:7) rho_est(11:12)];
        store_b_est_data(:,slot)=[user_present(1:2) user_present(6:7) user_present(11:12)].*[b_est(1:2) b_est(6:7) b_est(11:12)];
        store_d_est_data(:,slot)=[user_present(1:2) user_present(6:7) user_present(11:12)].*[d_est(1:2) d_est(6:7) d_est(11:12)];
        store_m_est_data(:,slot)=[user_present(1:2) user_present(6:7) user_present(11:12)].*[m_est(1:2) m_est(6:7) m_est(11:12)];
        
        for tmp1=[1 2 6 7 11 12]
            tmp=((ceil(tmp1/5)-1)*2 )+ (mod(tmp1,5)>=2)+1;
            max_quality_current_segment_being_downloaded(tmp,slot)=quality_QR_data(tmp1,total_number_of_segments_downloaded_by_users(tmp1)+1,6);
            i5_quality_current_segment_being_downloaded(tmp,slot)=quality_QR_data(tmp1,total_number_of_segments_downloaded_by_users(tmp1)+1,5);
            i4_quality_current_segment_being_downloaded(tmp,slot)=quality_QR_data(tmp1,total_number_of_segments_downloaded_by_users(tmp1)+1,4);
            i3_quality_current_segment_being_downloaded(tmp,slot)=quality_QR_data(tmp1,total_number_of_segments_downloaded_by_users(tmp1)+1,3);
            i2_quality_current_segment_being_downloaded(tmp,slot)=quality_QR_data(tmp1,total_number_of_segments_downloaded_by_users(tmp1)+1,2);
            min_quality_current_segment_being_downloaded(tmp,slot)=quality_QR_data(tmp1,total_number_of_segments_downloaded_by_users(tmp1)+1,1);
            store_quality_of_current_segment_being_downloaded(tmp,slot)=quality_of_current_segment_being_downloaded(tmp1);
            store_size_of_current_segment_being_downloaded(tmp,slot)=size_of_current_segment_being_downloaded(tmp1);
        end
    end
    %trace_info(:,((slot-1)*30+1):(slot*30))=[repmat(slot,1,30);1:30;b_base_est;b_enh_est;repmat(base_flag,1,30);...
    %    rate_alloc;ch_gain(:,slot)';total_base;...
    %    total_enh;size_of_current_segment_being_downloaded;...
    %    pending_base;pending_enh;qual_sel_q;qual_sel];
end
%dlmwrite('trace_imm.csv',trace_info)

%Evaluate quality metrics
mean_quality=zeros(1,num_users);
quality_std=zeros(1,num_users);
quality_change=zeros(1,num_users);
for user_index=1:num_users
    
    if user_index==11
        'bkpt';
    end
    load('Chao_model');
    tmp= tot_qual_matrix(1:min_seg_sim,user_index );
    tmpy=zeros(size(tmp));
    tmpx=chao_sigmoid(tmp-50,w(1),w(2),w(3),w(4));
    for t=1:min_seg_sim
        if t==1
            tmp=zeros(20,1);
        else
            tmp=[ tmpy( (t-1):-1: max((t-20),1)  );  zeros(max(0,20-t+1),1) ];
            %tmpy(t)=sum_{i=1}^20 a(i)*tmpy(t-i) + \sum_{i=1}^19 b(i)*tmpx(t-i);
        end
        tmpy(t)=sum(a.*tmp);
        if t==1
            tmp=zeros(19,1);
        else
            tmp=[ tmpx( (t-1):-1: max((t-19),1)  ) ; zeros(max(0,19-t+1),1) ];
            %tmpy(t)=sum_{i=1}^20 a(i)*tmpy(t-i) + \sum_{i=1}^19 b(i)*tmpx(t-i);
        end
        tmpy(t)=tmpy(t)+sum(b.*tmp);
    end
    tmpq=chao_sigmoid(tmpy,k(1),k(2),k(3),k(4));
    chao_metric(user_index)=mean(tmpq);
    
    mean_quality(user_index)=mean( tot_qual_matrix(1:min_seg_sim,user_index ));
    quality_std(user_index)=std( tot_qual_matrix(1:min_seg_sim,user_index ));
    quality_change(user_index)=sqrt(mean(diff( tot_qual_matrix(1:min_seg_sim,user_index )).^2));
    price_per_sec(user_index)=xi*mean( seg_size(1:min_seg_sim,user_index ))/t_seg;
end
QoE=mean_quality -  quality_std;
rebuf_time_est=(t_slot*slot)- total_enh*t_seg;
tmp=[chao_metric;QoE;mean_quality-quality_change;mean_quality;quality_std;quality_change;rebuf_time_est;count_rebuff_slots*t_slot;out_of_time_enh;total_allocated_rate/slot];
performance_metrics=[mean(tmp,2) (max(tmp')-min(tmp'))'./mean(tmp,2) tmp ];

write_data=[ [fnarg(1:5)'; variability_importance_constant;min_seg_sim;mean(beta_base_bar);fnarg(6);ch_gain_scale] performance_metrics];
dlmwrite(strcat('results',num2str(fnarg(7)),'.txt'),write_data,'-append')
dlmwrite(strcat('results',num2str(fnarg(7)),'.txt'),repmat(123123123,1,size(write_data,2)),'-append')


if 1 % to see traces of users' performance
    if traces_needed==1
        get_traces_for_users=[11];
        store_data_index=[1 2 0 0 0 3 4 0 0 0 5 6 0 0 0];
        for t_user_index=[get_traces_for_users]
            assert(store_data_index(t_user_index)>0);
            performance_metrics(:,t_user_index);
            subplot(2,1,1);plot(1:slot,store_b_est_data(store_data_index(t_user_index),1:slot));
            subplot(2,1,2);plot(1:slot,store_rho_est_data(store_data_index(t_user_index),1:slot));
            legend(strcat('User-11',num2str(store_data_index(t_user_index))))
            
            subplot(3,1,1);
            plot(1:slot,max_quality_current_segment_being_downloaded(store_data_index(t_user_index),1:slot),...
                1:slot,store_quality_of_current_segment_being_downloaded(store_data_index(t_user_index),1:slot),...
                1:slot,store_m_est_data(store_data_index(t_user_index),1:slot),...
                1:slot,min_quality_current_segment_being_downloaded(store_data_index(t_user_index),1:slot));
            subplot(3,1,2);plot(1:slot,store_b_est_data(store_data_index(t_user_index),1:slot));
            subplot(3,1,3);plot(1:slot,store_rho_est_data(store_data_index(t_user_index),1:slot));
        end
    end
end

if traces_needed==1
    % plot(1:slot,store_b_est_data(1,1:slot),1:slot,store_b_est_data(2,1:slot),1:slot,store_b_est_data(3,1:slot),...
    %     1:slot,store_b_est_data(4,1:slot),1:slot,store_b_est_data(5,1:slot),1:slot,store_b_est_data(6,1:slot));
    % legend('User-1','User-2','User-6','User-7','User-11','User-12')
    % figure
    plot(1:slot,store_d_est_data(1,1:slot),1:slot,store_d_est_data(2,1:slot),1:slot,store_d_est_data(3,1:slot),...
        1:slot,store_d_est_data(4,1:slot),1:slot,store_d_est_data(5,1:slot),1:slot,store_d_est_data(6,1:slot));
    legend('User-1','User-2','User-6','User-7','User-11','User-12')
end
%beep
returnstatus_INDEX=1;



function rate_allocation_RVBAR=RVBAR(ch_gain_cur_slot,base_queue)
global num_users rmin_RVBAR b_enh_est b_base_est user_present b_enh_init b_base_init
rate_allocation_RVBAR = zeros(1,num_users);
if sum(base_queue) ~= 0
    
    fn_b=(1/2)*max(b_base_est(base_queue > 0),0) + (1/2)*max(b_base_est(base_queue > 0)-...
        b_base_init(base_queue > 0)/2,0)+100*max(b_base_est(base_queue > 0)-b_base_init(base_queue > 0),0);
    
    %fn_b=fn_b.*fn_b_scale(base_queue > 0);
    
    ch_gain_cur_slot = ch_gain_cur_slot(base_queue > 0);
    users_active = user_present(base_queue > 0);
else
    fn_b=(1/2)*max(b_enh_est,0)+(1/2)*max(b_enh_est-...
        b_enh_init/2,0)+100*max(b_enh_est-b_enh_init,0);
    users_active = user_present;
end

if sum(rmin_RVBAR./ch_gain_cur_slot)>1
    input('Even minimum allocation is infeasible; Possible fixes: Reduce min allocation, or Increase average channel quality')
end

tmp=((fn_b.*ch_gain_cur_slot')==max(fn_b.*ch_gain_cur_slot'));
tmp=tmp*(1-(sum( users_active'.*(rmin_RVBAR./ch_gain_cur_slot) )))/sum(tmp);
rate_allocation_RVBAR_tmp=ch_gain_cur_slot.*tmp'+rmin_RVBAR*users_active';
rate_allocation_RVBAR_tmp=rate_allocation_RVBAR_tmp';
if sum(base_queue) ~= 0
    rate_allocation_RVBAR(base_queue > 0) = rate_allocation_RVBAR_tmp;
else
    rate_allocation_RVBAR = rate_allocation_RVBAR_tmp;
end
%assert(abs(sum(rate_allocation_RVBAR_tmp'./ch_gain_cur_slot)-1) <.0001)

function rate_allocation_PF=PF_rate_allocation(ch_gain_cur_slot,base_queue)
global num_users rmin_RVBAR user_present rate_allocation_PF_tmp
rate_allocation_PF = zeros(1,num_users);

if sum(base_queue) ~= 0
    ch_gain_cur_slot = ch_gain_cur_slot(base_queue > 0);
    users_active = user_present(base_queue > 0);
else
    ch_gain_cur_slot = ch_gain_cur_slot;
    users_active = user_present;
end
check_feasibility_of_closed_form_solution=ch_gain_cur_slot'/sum(users_active);%this is the closed form solution
if min(check_feasibility_of_closed_form_solution)>=rmin_RVBAR
    rate_allocation_PF_tmp=check_feasibility_of_closed_form_solution;
elseif ~isempty(check_feasibility_of_closed_form_solution)
    if sum(rmin_RVBAR./ch_gain_cur_slot)>1
        input('Even minimum allocation is infeasible; Possible fixes: Reduce min allocation, or Increase average channel quality')
    end
    'cvx running'
    cvx_begin
    cvx_quiet(true);
    variable rate_allocation_PF_tmp(1,users_active);
    maximize( sum(log(rate_allocation_PF_tmp ) ) );
    subject to
    sum(rate_allocation_PF_tmp./ch_gain_cur_slot')<=1;
    rate_allocation_PF_tmp >= rmin_RVBAR*ones(1,sum(users_active));
    cvx_end
else
    rate_allocation_PF_tmp = 0;
end
if sum(base_queue) ~= 0
    rate_allocation_PF(base_queue > 0) = rate_allocation_PF_tmp;
else
    rate_allocation_PF = rate_allocation_PF_tmp;
end


function rate_allocation_PFL=PF_long_run_rate_allocation(ch_gain_cur_slot)
global rho_est rmin_RVBAR

if sum(rmin_RVBAR./ch_gain_cur_slot)>1
    input('Even minimum allocation is infeasible; Possible fixes: Reduce min allocation, or Increase average channel quality')
end

allocation_metric=  (ch_gain_cur_slot'./(rho_est))  ;
tmp=(allocation_metric==max(allocation_metric));

tmp=tmp*(1-(sum(rmin_RVBAR./ch_gain_cur_slot)))/sum(tmp);
rate_allocation_PFL=ch_gain_cur_slot.*tmp'+rmin_RVBAR;
rate_allocation_PFL=rate_allocation_PFL';
assert(abs(sum(rate_allocation_PFL'./ch_gain_cur_slot)-1) <.0001)


function rate_allocation_S=sarabjot_rate_allocation(ch_gain_cur_slot)
global num_users rho_est rmin_RVBAR count_rebuff_slots slot total_number_of_segments_downloaded_by_users t_seg t_slot size_of_current_segment_being_downloaded

if sum(rmin_RVBAR./ch_gain_cur_slot)>1
    input('Even minimum allocation is infeasible; Possible fixes: Reduce min allocation, or Increase average channel quality')
end

V=ones(1,num_users);
tmp=count_rebuff_slots/slot;
if sum(tmp)>0
    V=V+(num_users*tmp./sum(tmp));
end

if slot>10000
    'bkpt';
end

fmin=30;%so as to aim for 1 sec buffer level %this value does not change performance much; even 200 does not
f=30*max(total_number_of_segments_downloaded_by_users*t_seg-(t_slot*slot),0); %multiplied by 30 assuming frame rate of 30 fps
S_alpha=1;S_beta=1;
allocation_metric1= S_alpha* (ch_gain_cur_slot'./(size_of_current_segment_being_downloaded/30)).*exp(S_beta*(fmin-f))    ;
allocation_metric2=  (ch_gain_cur_slot'./rho_est)  ;
allocation_metric= V.* ( allocation_metric1 + allocation_metric2  );

tmp=(allocation_metric==max(allocation_metric))';
tmp=tmp*(1-(sum(rmin_RVBAR./ch_gain_cur_slot)))/sum(tmp);
rate_allocation_S=ch_gain_cur_slot.*tmp+rmin_RVBAR;
rate_allocation_S=rate_allocation_S';
assert(abs(sum(rate_allocation_S'./ch_gain_cur_slot)-1) <.0001)


function rate_allocation_PF=weighted_PF_rate_allocation(ch_gain_cur_slot)
global num_users rmin_RVBAR
weights=[3 3 3 3 3 2 2 2 2 2 1 1 1 1 1];
check_feasibility_of_closed_form_solution=ch_gain_cur_slot'/num_users;%this is the closed form solution
if min(check_feasibility_of_closed_form_solution)>=rmin_RVBAR
    rate_allocation_PF=check_feasibility_of_closed_form_solution;
else
    if sum(rmin_RVBAR./ch_gain_cur_slot)>1
        input('Even minimum allocation is infeasible; Possible fixes: Reduce min allocation, or Increase average channel quality')
    end
    cvx_begin
    cvx_quiet(true);
    variable rate_allocation_PF(1,num_users);
    maximize( sum(log(rate_allocation_PF ) ) );
    subject to
    sum(rate_allocation_PF./ch_gain_cur_slot')<=1;
    rate_allocation_PF >= rmin_RVBAR*ones(1,num_users);
    cvx_end
end


function y=chao_sigmoid(x,a,b,c,d)
y=d./(1+exp(-(a*x+b)))+c;

