%important_constant_1: importance of variability;
%discretization will cause following problems:
%if too large will not allow increase in quality even when no rebuffering issues

%important_constant_2
%decides the sensitivity of corrections in b_est

function return_status_INDEX=runsim_orig_test(fnarg)

global num_users rmin_RVBAR m_est v_est b_est rho_est sigma_est epsilon max_base fn_b_scale user_present...
    count_rebuff_slots slot total_number_of_segments_downloaded_by_users...
    segment_duration beta_bar fn_b_offset slot_duration size_of_current_segment_being_downloaded enh_layer

%Amir: Take the input vector fnarg and assign values for respective variables.
%close all;clc;
min_num_of_segments_to_simulate_for_each_user=1200;num_users=fnarg(1);
rate_allocation_scheme=fnarg(2);quality_adaptation_scheme=fnarg(3);
homogeneous_channels=fnarg(4);markov_channels=fnarg(5);
pbar_val=fnarg(6);max_base=fnarg(8);

enh_layer = 6;
segment_duration=1;%in seconds
slot_duration=.01;%in seconds

LPF_run_slots_before_using_RVBAR=0;%needed to ensure that there is no rebuffering in the beginning for the weakest users due to poor allocation
segments_to_run_greedy_before_QVBAR=0;
variability_importance_constant=0.1; 
count_rebuff_slots=zeros(1,num_users);
user_present=ones(1,num_users);


%VBAR parameters

epsilon=0.05;
beta_bar = -0.2*ones(1,num_users);
fn_b_scale = 0.2*ones(1,num_users);
fn_b_offset = 0*ones(1,num_users);
b_init = 15*epsilon*ones(1,num_users);
total_allocated_rate = 0*ones(1,num_users);

size_of_current_segment_being_downloaded=100*ones(1,num_users);
quality_of_current_segment_being_downloaded=20*ones(1,num_users);
rmin_RVBAR=.001;
xi=.01;%Amir: Unit cost for data transmission (p_i^d)

%parameter_intialization
m_est=25*ones(1,num_users);
v_est=5*ones(1,num_users);
b_est=b_init;

sigma_est = 100*ones(1,num_users);
rho_est = 100*ones(1,num_users);


tmp='generate_quality_function_data/data/quality_QR_data_N=';
tmp=strcat(tmp,num2str(33),'_T=',num2str(4500));
quality_QR_data=dlmread(tmp);
quality_QR_data=100-quality_QR_data;
quality_QR_data=reshape(quality_QR_data,33,4500,6);
quality_QR_data=quality_QR_data(   [1:(0+(num_users/3)) 12:(11+(num_users/3)) 23:(22+(num_users/3))]          ,:,:);
%quality_QR_data = quality_QR_data(:,:,1:enh_layer);
quality_QR_data = repmat(quality_QR_data,1,3);

tmp='generate_quality_function_data/data/rate_QR_data_N=';
tmp=strcat(tmp,num2str(33),'_T=',num2str(4500));
rate_QR_data=dlmread(tmp);
rate_QR_data=reshape(rate_QR_data,33,4500,6);
rate_QR_data=rate_QR_data(   [1:(0+(num_users/3)) 12:(11+(num_users/3)) 23:(22+(num_users/3))]          ,:,:);
%rate_QR_data = rate_QR_data(:,:,1:enh_layer);
rate_QR_data = repmat(rate_QR_data,1,3);


%%%%%%% for testing %%%%%%%%%
%quality_QR_data=sqrt(rate_QR_data);
%%%%%%%

%tmp='generate_channel_data_code/data/new1_chdata_N=';
tmp='generate_channel_data_code/data/corr_5/chdata_N=';
tmp=strcat(tmp,num2str(33),'_T=',num2str(150000),'_round=',num2str(fnarg(9)));
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
ch_gain=ch_gain_scale*slot_duration*dlmread(tmp)+30*rmin_RVBAR;
tmp=size(ch_gain,1)*size(ch_gain,2);

ch_gain=(reshape(ch_gain,tmp/33,33))';
ch_gain=ch_gain(    [1:(0+(num_users/3)) 12:(11+(num_users/3)) 23:(22+(num_users/3))]        , : ) ;
ch_gain = repmat(ch_gain,1,3);

downloaded_segments_quality_data_matrix=zeros(min_num_of_segments_to_simulate_for_each_user+1,num_users);
downloaded_segments_size_data_matrix=zeros(min_num_of_segments_to_simulate_for_each_user+1,num_users);

rebuf_dist = zeros(num_users,150000);
buffer_fill = zeros(num_users,150000);
virtual_b = zeros(num_users,150000);
qual_index = ones(num_users,min_num_of_segments_to_simulate_for_each_user);

%intitalization
slot=0;%index for the current slot
total_number_of_segments_downloaded_by_users = -1*ones(1,num_users);

qual_sel = zeros(1,num_users);
qual_sel_q = zeros(1,num_users);
pending_amount_of_data_for_current_segment=downloaded_segments_size_data_matrix(1,:);
%trace_info = zeros(10,100000*3);
first_hit = zeros(1,num_users);

while min(total_number_of_segments_downloaded_by_users)<=min_num_of_segments_to_simulate_for_each_user
    
    buffer_occupancy = (total_number_of_segments_downloaded_by_users * segment_duration) - ((slot-count_rebuff_slots) * slot_duration);
    active = find(buffer_occupancy < max_base);
    first_hit(first_hit==0 & buffer_occupancy >= max_base) = b_est(first_hit == 0 & buffer_occupancy >= max_base);
    
    slot=slot+1;  
    buffer_fill(:,slot) = buffer_occupancy;
    rebuf_dist(:,slot) = ((((slot-count_rebuff_slots)*slot_duration/segment_duration)- total_number_of_segments_downloaded_by_users)-3>0);
    virtual_b(:,slot) = b_est;
    
    %rate allocation code goes here
    if rate_allocation_scheme==0 %RVBAR
        if slot<LPF_run_slots_before_using_RVBAR %Amir:Use proportional fairness.
            cur_slot_rate_allocation=PF_long_run_rate_allocation(ch_gain(:,slot));
        else
            cur_slot_rate_allocation=RVBAR(ch_gain(:,slot),active);
            cur_slot_rate_allocation=cur_slot_rate_allocation';
        end
        epsilon_slot=max(.001,1/slot);
    elseif rate_allocation_scheme==1%PF scheme (slot level fairness)
        cur_slot_rate_allocation=PF_rate_allocation(ch_gain(:,slot),active)';
        epsilon_slot=.1;
    elseif rate_allocation_scheme==2 %PF with long term rate fairness
        cur_slot_rate_allocation=PF_long_run_rate_allocation(ch_gain(:,slot));
        epsilon_slot=.01;
    end
    
    total_allocated_rate=total_allocated_rate+cur_slot_rate_allocation;
    rho_est(active)=rho_est(active)+epsilon_slot*( (cur_slot_rate_allocation(active)/slot_duration) - rho_est(active));
    
    b_est(active) = b_est(active) + epsilon*slot_duration./(segment_duration.*(1+beta_bar(active))); 
    %b_est = b_est + epsilon*slot_duration./(segment_duration.*(1+beta_bar)); 
    
    %According to code, the allocated rate is in units of bits!
    pending_amount_of_data_for_current_segment=pending_amount_of_data_for_current_segment-cur_slot_rate_allocation;
    set_of_users_who_finish_segment_transmission_in_current_slot=find(pending_amount_of_data_for_current_segment<0);
    
    %This whole while loop is only for those users who have
    %completely downloaded their latest segment.
    while(isempty(set_of_users_who_finish_segment_transmission_in_current_slot)==0)
        user_index=set_of_users_who_finish_segment_transmission_in_current_slot(1);
        total_number_of_segments_downloaded_by_users(user_index)=total_number_of_segments_downloaded_by_users(user_index)+1;
        
        %change the next line if quality metric knowledge is imperfect
        q_vals_for_this_user_latest_segment=reshape(quality_QR_data(user_index,total_number_of_segments_downloaded_by_users(user_index)+1,:),1,enh_layer);
        sigma_vals_for_this_user_latest_segment=reshape(rate_QR_data(user_index,total_number_of_segments_downloaded_by_users(user_index)+1,:),1,enh_layer);
        
        if quality_adaptation_scheme==-1 %scheme for testing code
            quality_choice_for_this_user=unidrnd(5);segment_size_for_quality_choice=quality_choice_for_this_user^2;
        elseif quality_adaptation_scheme==0 %QVBAR
            if total_number_of_segments_downloaded_by_users(user_index)<segments_to_run_greedy_before_QVBAR
                selected_representation_index=max(1,sum((sigma_vals_for_this_user_latest_segment<=((1+beta_bar(user_index))*rho_est(user_index)))));
                quality_choice_for_this_user=q_vals_for_this_user_latest_segment(selected_representation_index);
                segment_size_for_quality_choice=sigma_vals_for_this_user_latest_segment(selected_representation_index);
            else
                 fn_b=fn_b_scale(user_index)*((1/2)*max(b_est(user_index),0)+(1/2)*max(b_est(user_index)-b_init(user_index)/2,0)...
                        +100*max(b_est(user_index)-b_init(user_index),0));
                 tmp_metric= q_vals_for_this_user_latest_segment ...
                        - (fn_b/(1+beta_bar(user_index))*segment_duration).*sigma_vals_for_this_user_latest_segment ...
                        - variability_importance_constant * ( ( q_vals_for_this_user_latest_segment - m_est(user_index)).^2 );
             
                %fn_b=b_est(user_index);
                %fn_b=fn_b * fn_b_scale(user_index);
                %tmp_metric = ( q_vals_for_this_user_latest_segment ...
                %    - fn_b/((1+beta_bar(user_index)) * segment_duration) .* sigma_vals_for_this_user_latest_segment) ...
                %    - variability_importance_constant * ( q_vals_for_this_user_latest_segment - m_est(user_index)).^2 ;
                 
                %fn_b=fn_b_scale(user_index)*(b_est(user_index) + max((b_est(user_index) - 50),0)^2);
                %tmp_metric=( q_vals_for_this_user_latest_segment ...
                %    - (fn_b/((1+beta_bar(user_index))*segment_duration)).*sigma_vals_for_this_user_latest_segment) ...
                %   - variability_importance_constant*(( q_vals_for_this_user_latest_segment - m_est(user_index)).^2 );
                
                [~, selected_representation_index]=max(tmp_metric);
                qual_sel(user_index) = selected_representation_index;
               
                qual_index(user_index,total_number_of_segments_downloaded_by_users(user_index)+1) = selected_representation_index;
                quality_choice_for_this_user=q_vals_for_this_user_latest_segment(selected_representation_index);
                qual_sel_q(user_index) = quality_choice_for_this_user;
                segment_size_for_quality_choice=sigma_vals_for_this_user_latest_segment(selected_representation_index);
            end
        elseif quality_adaptation_scheme==1%greedy segment selection based on current estimate of throughput
            selected_representation_index=max(1,sum((sigma_vals_for_this_user_latest_segment<=((1-.01)*rho_est(user_index)))));
            quality_choice_for_this_user=q_vals_for_this_user_latest_segment(selected_representation_index);
            segment_size_for_quality_choice=sigma_vals_for_this_user_latest_segment(selected_representation_index);
        end
        
        
        
        epsilon_seg=epsilon;
        m_est(user_index)=m_est(user_index) + epsilon_seg*( quality_choice_for_this_user- m_est(user_index) );
        v_est(user_index)=v_est(user_index) + epsilon_seg*( (quality_choice_for_this_user - m_est(user_index))^2 - v_est(user_index)  );
        sigma_est(user_index) = sigma_est(user_index) + epsilon_seg*(segment_size_for_quality_choice -  sigma_est(user_index));
        %b_est(user_index)=max(max(b_est(user_index)-epsilon*segment_duration,b_init(user_index) - epsilon*max_base),0);
        
        if buffer_occupancy(user_index) < max_base - 1
            b_est(user_index)=max(b_est(user_index) - epsilon*segment_duration,0);
            first_hit(user_index) = 0;
        elseif buffer_occupancy(user_index) >= max_base - 1 && first_hit(user_index) ~= 0
            b_est(user_index) = first_hit(user_index);
        end
        
        downloaded_segments_quality_data_matrix(total_number_of_segments_downloaded_by_users(user_index)+1,user_index)=...
            quality_choice_for_this_user;
        downloaded_segments_size_data_matrix(total_number_of_segments_downloaded_by_users(user_index)+1,user_index)=...
            segment_size_for_quality_choice;
        
        size_of_current_segment_being_downloaded(user_index)=segment_size_for_quality_choice;
        quality_of_current_segment_being_downloaded(user_index)=quality_choice_for_this_user;
        
        pending_amount_of_data_for_current_segment(user_index)=pending_amount_of_data_for_current_segment(user_index)+segment_size_for_quality_choice;
        set_of_users_who_finish_segment_transmission_in_current_slot=find(pending_amount_of_data_for_current_segment<0);
    end
    count_rebuff_slots=count_rebuff_slots + ((((slot-count_rebuff_slots)*slot_duration/segment_duration)- total_number_of_segments_downloaded_by_users)-3>0);
    
    %trace_info(:,((slot-1)*3+1):(slot*3))=[repmat(slot,1,3);1:3;b_est;...
    %    cur_slot_rate_allocation;ch_gain(:,slot)';total_number_of_segments_downloaded_by_users;...
    %    size_of_current_segment_being_downloaded;...
    %    pending_amount_of_data_for_current_segment;qual_sel_q;qual_sel];

end

%dlmwrite('trace_orig.csv',trace_info);
%Evaluate quality metrics
mean_quality=zeros(1,num_users);
quality_std=zeros(1,num_users);
quality_change=zeros(1,num_users);
for user_index=1:num_users
  
    load('Chao_model');
    tmp= downloaded_segments_quality_data_matrix(1:min_num_of_segments_to_simulate_for_each_user,user_index );
    tmpy=zeros(size(tmp));
    tmpx=chao_sigmoid(tmp-50,w(1),w(2),w(3),w(4));
    for t=1:min_num_of_segments_to_simulate_for_each_user
        if t==1
            tmp=zeros(20,1);
        else
            tmp=[ tmpy( (t-1):-1: max((t-20),1)  );  zeros(max(0,20-t+1),1) ];
        end
        tmpy(t)=sum(a.*tmp);
        if t==1
            tmp=zeros(19,1);
        else
            tmp=[ tmpx( (t-1):-1: max((t-19),1)  ) ; zeros(max(0,19-t+1),1) ];
        end
        tmpy(t)=tmpy(t)+sum(b.*tmp);
    end
    tmpq=chao_sigmoid(tmpy,k(1),k(2),k(3),k(4));
    chao_metric(user_index)=mean(tmpq);
    
    mean_quality(user_index)=mean( downloaded_segments_quality_data_matrix(1:min_num_of_segments_to_simulate_for_each_user,user_index ));
    quality_std(user_index)=std( downloaded_segments_quality_data_matrix(1:min_num_of_segments_to_simulate_for_each_user,user_index ));
    quality_change(user_index)=sqrt(mean(diff( downloaded_segments_quality_data_matrix(1:min_num_of_segments_to_simulate_for_each_user,user_index )).^2));
    price_per_sec(user_index)=xi*mean( downloaded_segments_size_data_matrix(1:min_num_of_segments_to_simulate_for_each_user,user_index ))/segment_duration;
end
QoE=mean_quality -  quality_std;
rebuf_time_est=(slot_duration*slot)- total_number_of_segments_downloaded_by_users*segment_duration;
tmp=[chao_metric;QoE;mean_quality-quality_change;mean_quality;quality_std;quality_change;rebuf_time_est;count_rebuff_slots*slot_duration;price_per_sec;total_allocated_rate/slot];

performance_metrics=[mean(tmp,2) (max(tmp')-min(tmp'))'./mean(tmp,2) tmp ];
write_data=[ [fnarg(1:5)'; variability_importance_constant;min_num_of_segments_to_simulate_for_each_user;mean(beta_bar);fnarg(6);ch_gain_scale] performance_metrics];
size(write_data,2);
dlmwrite(strcat('results',num2str(fnarg(7)),'.txt'),write_data,'-append');
dlmwrite(strcat('results',num2str(fnarg(7)),'.txt'),repmat(123123123,1,size(write_data,2)),'-append');

%dlmwrite('rebuf_dist.txt',rebuf_dist(:,1:slot));
dlmwrite('buffer_fill.txt',buffer_fill(:,1:50000));
dlmwrite('virtual_buffer.txt',virtual_b(:,1:50000));
%dlmwrite('quality_st.txt',qual_index(:,1:min_num_of_segments_to_simulate_for_each_user));
returnstatus_INDEX=1;

function rate_allocation_RVBAR=RVBAR(ch_gain_cur_slot,active)
global   rmin_RVBAR b_est user_present num_users fn_b_scale fn_b_offset

rate_allocation_RVBAR = zeros(num_users,1);                         
if sum(rmin_RVBAR./ch_gain_cur_slot(active))>1
    input('Even minimum allocation is infeasible; Possible fixes: Reduce min allocation, or Increase average channel quality')
end

%fn_b=h_0*(b_est(active)/epsilon + max((b_est(active) - 20)/epsilon,0).^2);
fn_b=fn_b_scale(active).*max(b_est(active)-fn_b_offset(active),0).*user_present(active);
tmp=((fn_b.*ch_gain_cur_slot(active)')==max(fn_b.*ch_gain_cur_slot(active)'));
tmp=tmp*(1-(sum( user_present(active)'.*(rmin_RVBAR./ch_gain_cur_slot(active)) )))/sum(tmp);
rate_allocation_tmp=ch_gain_cur_slot(active).*tmp'+rmin_RVBAR*user_present(active)';
rate_allocation_RVBAR(active) = rate_allocation_tmp';

function rate_allocation_PF=PF_rate_allocation(ch_gain_cur_slot,active)
global num_users rmin_RVBAR rate_allocation_tmp
rate_allocation_PF = zeros(num_users,1);
if ~isempty(active)
    check_feasibility_of_closed_form_solution=ch_gain_cur_slot(active)/num_users;%this is the closed form solution
    if min(check_feasibility_of_closed_form_solution)>=rmin_RVBAR
        rate_allocation_tmp=check_feasibility_of_closed_form_solution;
    else
        if sum(rmin_RVBAR./ch_gain_cur_slot(active))>1
            input('Even minimum allocation is infeasible; Possible fixes: Reduce min allocation, or Increase average channel quality')
        end
        'cvx';
        cvx_begin
        cvx_quiet(true);
        variable rate_allocation_tmp(1,length(active));
        maximize( sum(log(rate_allocation_tmp) ) );
        subject to
        sum(rate_allocation_tmp./ch_gain_cur_slot(active))<=1;
        rate_allocation_tmp >= rmin_RVBAR * ones(1,length(active));
        cvx_end
    end
    rate_allocation_PF(active) = rate_allocation_tmp';
end

function rate_allocation_PFL=PF_long_run_rate_allocation(ch_gain_cur_slot)
global rho_est rmin_RVBAR

if sum(rmin_RVBAR./ch_gain_cur_slot)>1
    ch_gain_cur_slot;
    input('Even minimum allocation is infeasible; Possible fixes: Reduce min allocation, or Increase average channel quality')
end

allocation_metric=  (ch_gain_cur_slot'./(rho_est))  ;
tmp=(allocation_metric==max(allocation_metric));

tmp=tmp*(1-(sum(rmin_RVBAR./ch_gain_cur_slot)))/sum(tmp);
rate_allocation_PFL=ch_gain_cur_slot.*tmp'+rmin_RVBAR;
rate_allocation_PFL=rate_allocation_PFL';
assert(abs(sum(rate_allocation_PFL'./ch_gain_cur_slot)-1) <.0001)

function y=chao_sigmoid(x,a,b,c,d)
y=d./(1+exp(-(a*x+b)))+c;








