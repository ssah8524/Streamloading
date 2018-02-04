%important_constant_1: importance of variability; 
%discretization will cause following problems:
%if too large will not allow increase in quality even when no rebuffering issues

%important_constant_2
%decides the sensitivity of corrections in b_est




function return_status_INDEX=original(fnarg)
global num_users eta_RVBAR rmin_RVBAR m_est v_est b_est d_est rho_est sigma_est epsilon BS_b_est fn_b_scale user_present...
    fn_b_offset count_rebuff_slots slot total_number_of_segments_downloaded_by_users segment_duration slot_duration size_of_current_segment_being_downloaded
    
close all;clc;min_num_of_segments_to_simulate_for_each_user=600;num_users=fnarg(1);
rate_allocation_scheme=fnarg(2);quality_adaptation_scheme=fnarg(3);
homogeneous_channels=fnarg(4);markov_channels=fnarg(5);
pbar_val=fnarg(6);

segment_duration=1;%in seconds
slot_duration=.01;%in seconds
initial_num_segments_to_be_downloaded_at_low_quality=0;
LPF_run_slots_before_using_RVBAR=0;%needed to ensure that there is no rebuffering in the beginning for the weakest users due to poor allocation
segments_to_run_greedy_before_QVBAR=0;
variability_importance_constant=.1;%important_constant_1

count_rebuff_slots=zeros(1,num_users);
user_present=ones(1,num_users);

%VBAR parameters
beta_bar=(0)*ones(1,num_users);%-(1/21)*ones(1,num_users);
fn_b_scale=0.01*ones(1,num_users);%0.01*ones(1,num_users);
fn_d_scale=10*ones(1,num_users);
fn_b_offset=0*ones(1,num_users);
b_init=15*ones(1,num_users);
d_init=1*ones(1,num_users);
total_allocated_rate=0*ones(1,num_users);
size_of_current_segment_being_downloaded=100*ones(1,num_users);
quality_of_current_segment_being_downloaded=20*ones(1,num_users);

epsilon=0.05;
eta_RVBAR =.01;
rmin_RVBAR=.001;
xi=.01;
pbar=pbar_val*ones(1,num_users);
delta_square_root_approx=0.01;

%parameter_intialization
m_est=20*ones(1,num_users);
v_est=5*ones(1,num_users);
b_est=b_init;
d_est=d_init;
sigma_est=100*ones(1,num_users);
rho_est=100*ones(1,num_users);


tmp='generate_quality_function_data/data/quality_QR_data_N=';
tmp=strcat(tmp,num2str(33),'_T=',num2str(4500));
quality_QR_data=dlmread(tmp);
quality_QR_data=100-quality_QR_data;
quality_QR_data=reshape(quality_QR_data,33,4500,6);
quality_QR_data=quality_QR_data(   [1:(0+(num_users/3)) 12:(11+(num_users/3)) 23:(22+(num_users/3))]          ,:,:);


use_continuous_curves_approximation=0;
if use_continuous_curves_approximation==1
   'Using continuous_curves_approximation' 
end

if 0%to try constant QR curves
    'to try constant QR curves'
    for user_index=1:num_users
        q= reshape(quality_QR_data(user_index,111,:),1,6);
        r= reshape(rate_QR_data(user_index,111,:),1,6);
        quality_QR_data(user_index,:,:)=reshape(repmat(q,4500,1),1,4500,6);
        rate_QR_data(user_index,:,:)=reshape(repmat(r,4500,1),1,4500,6);
        hold on;
        plot(r,q);
    end
    
   clear q;clear r; 
end

tmp='generate_quality_function_data/data/rate_QR_data_N=';
tmp=strcat(tmp,num2str(33),'_T=',num2str(4500));
rate_QR_data=dlmread(tmp);
rate_QR_data=reshape(rate_QR_data,33,4500,6);
rate_QR_data=rate_QR_data(   [1:(0+(num_users/3)) 12:(11+(num_users/3)) 23:(22+(num_users/3))]          ,:,:);


%%%%%%% for testing %%%%%%%%%
%quality_QR_data=sqrt(rate_QR_data);
%%%%%%%




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
ch_gain=ch_gain_scale*slot_duration*dlmread(tmp)+30*rmin_RVBAR;
tmp=size(ch_gain,1)*size(ch_gain,2);

ch_gain=(reshape(ch_gain,tmp/33,33))';


ch_gain=ch_gain(    [1:(0+(num_users/3)) 12:(11+(num_users/3)) 23:(22+(num_users/3))]        , : ) ;

traces_needed=0;
if traces_needed==1
    store_m_est_data=zeros(6,size(ch_gain,2));
    store_b_est_data=zeros(6,size(ch_gain,2));
    store_d_est_data=zeros(6,size(ch_gain,2));
    store_rho_est_data=zeros(6,size(ch_gain,2));
    store_quality_of_current_segment_being_downloaded=zeros(6,size(ch_gain,2));
    store_quality_of_current_segment_being_downloaded=zeros(6,size(ch_gain,2));
    max_quality_current_segment_being_downloaded=zeros(6,size(ch_gain,2));
    i5_quality_current_segment_being_downloaded=zeros(6,size(ch_gain,2));
    i4_quality_current_segment_being_downloaded=zeros(6,size(ch_gain,2));
    i3_quality_current_segment_being_downloaded=zeros(6,size(ch_gain,2));
    i2_quality_current_segment_being_downloaded=zeros(6,size(ch_gain,2));
    min_quality_current_segment_being_downloaded=zeros(6,size(ch_gain,2));
end

downloaded_segments_quality_data_matrix=zeros(min_num_of_segments_to_simulate_for_each_user+1,num_users);
downloaded_segments_size_data_matrix=zeros(min_num_of_segments_to_simulate_for_each_user+1,num_users);

%intitalization
slot=0;%index for the current slot
total_number_of_segments_downloaded_by_users=-1*ones(1,num_users);

% tmpq=quality_QR_data(1:num_users,1,1);
% tmpq=reshape(tmpq,1,num_users);%users start with lowest quality
% tmpr=rate_QR_data(1:num_users,1,1);
% tmpr=reshape(tmpr,1,num_users);%users start with compression rate corresponding to lowest quality
% downloaded_segments_quality_data_matrix(1,:)=tmpq;%users start with lowest quality
% downloaded_segments_size_data_matrix(1,:)=tmpr;
% sigma_est=tmpr;

pending_amount_of_data_for_current_segment=downloaded_segments_quality_data_matrix(1,:);
while min(total_number_of_segments_downloaded_by_users)<=min_num_of_segments_to_simulate_for_each_user
    slot=slot+1;
    
    %beta_bar=min(-1/21+max(slot-(10^4),0)*(10^-4)/51,-1/51)*ones(1,num_users);
    
    %to check transient, include following lines
    if 0
        if 0 %introduce three new users at 3 different times
            need_init_for_new_user=0;
            if slot==1
                user_present(1)=0;user_present(6)=0;user_present(11)=0;
            end
            if slot==1*10^4
                need_init_for_new_user=1;
                user_index_to_init=1;
            elseif slot==2*10^4
                need_init_for_new_user=1;
                user_index_to_init=6;
            elseif slot==3*10^4
                need_init_for_new_user=1;
                user_index_to_init=11;
            end
            if need_init_for_new_user==1
                user_present(user_index_to_init)=1;
                m_est(user_index_to_init)=30;
                v_est(user_index_to_init)=5;
                b_est(user_index_to_init)=b_init(user_index_to_init);
                d_est(user_index_to_init)=d_init(user_index_to_init);
                sigma_est(user_index_to_init)=100;
                rho_est(user_index_to_init)=100;
            end
        end
        
        if 1 %remove 3 users
            need_init_for_new_user=0;
            if slot==1*10^4
                need_init_for_new_user=1;
                user_index_to_init=1;
            elseif slot==1*10^4
                need_init_for_new_user=1;
                user_index_to_init=6;
            elseif slot==1*10^4
                need_init_for_new_user=1;
                user_index_to_init=11;
            end
            if need_init_for_new_user==1
                user_present(user_index_to_init)=0;
            end
        end
    end
    
    %rate allocation code goes here
    if rate_allocation_scheme==-1
        cur_slot_rate_allocation=10*rand(1,num_users);%scheme for testing code
    elseif rate_allocation_scheme==0%RVBAR
        if slot<LPF_run_slots_before_using_RVBAR
            cur_slot_rate_allocation=PF_long_run_rate_allocation(ch_gain(:,slot));
        else
            cur_slot_rate_allocation=RVBAR(ch_gain(:,slot));
            cur_slot_rate_allocation=cur_slot_rate_allocation';            
        end
        epsilon_slot=max(.001,1/slot);
    elseif rate_allocation_scheme==1%PF scheme (slot level fairness)
        cur_slot_rate_allocation=PF_rate_allocation(ch_gain(:,slot));
        epsilon_slot=.1;
    elseif rate_allocation_scheme==2%weighted_PF scheme
        cur_slot_rate_allocation=weighted_PF_rate_allocation(ch_gain(:,slot));
        epsilon_slot=.1;
    elseif rate_allocation_scheme==3 %PF with long term rate fairness
        cur_slot_rate_allocation=PF_long_run_rate_allocation(ch_gain(:,slot));
        epsilon_slot=.01;
    elseif rate_allocation_scheme==4 %Sarabjot's scheme
        cur_slot_rate_allocation=sarabjot_rate_allocation(ch_gain(:,slot));
        epsilon_slot=.01;
    end
    
    total_allocated_rate=total_allocated_rate+cur_slot_rate_allocation;
    rho_est=rho_est+epsilon_slot*( (cur_slot_rate_allocation/slot_duration) - rho_est);  
    
    b_est=b_est+slot_duration./(segment_duration.*(1+beta_bar));    
    
    pending_amount_of_data_for_current_segment=pending_amount_of_data_for_current_segment-cur_slot_rate_allocation;
    set_of_users_who_finish_segment_transmission_in_current_slot=find(pending_amount_of_data_for_current_segment<0);    
       
    while(isempty(set_of_users_who_finish_segment_transmission_in_current_slot)==0)
        user_index=set_of_users_who_finish_segment_transmission_in_current_slot(1);
        total_number_of_segments_downloaded_by_users(user_index)=total_number_of_segments_downloaded_by_users(user_index)+1;
        
        if mod(sum(total_number_of_segments_downloaded_by_users)+16,200)==0
            round([slot/100 total_number_of_segments_downloaded_by_users;mean(count_rebuff_slots*.01) count_rebuff_slots*.01])            
            'bkpt';
        end
        
        %change the next line if quailty metric knowledge is imperfect
        q_vals_for_this_user_latest_segment=reshape(quality_QR_data(user_index,total_number_of_segments_downloaded_by_users(user_index)+1,:),1,6);
        sigma_vals_for_this_user_latest_segment=reshape(rate_QR_data(user_index,total_number_of_segments_downloaded_by_users(user_index)+1,:),1,6);
        
        if total_number_of_segments_downloaded_by_users(user_index)<initial_num_segments_to_be_downloaded_at_low_quality
            quality_choice_for_this_user=q_vals_for_this_user_latest_segment(1);%users start with lowest quality
            segment_size_for_quality_choice=sigma_vals_for_this_user_latest_segment(1);%users start with compression rate corresponding to lowest quality
        else
            %pick the next segment quality: quality adaptation code goes here            
            min(.1*total_number_of_segments_downloaded_by_users(user_index),10);
            
            if quality_adaptation_scheme==-1 %scheme for testing code
                quality_choice_for_this_user=unidrnd(5);segment_size_for_quality_choice=quality_choice_for_this_user^2;
            elseif quality_adaptation_scheme==0 %QVBAR
                if total_number_of_segments_downloaded_by_users(user_index)<segments_to_run_greedy_before_QVBAR
                    selected_representation_index=max(1,sum((sigma_vals_for_this_user_latest_segment<=((1+beta_bar(user_index))*rho_est(user_index)))));
                    quality_choice_for_this_user=q_vals_for_this_user_latest_segment(selected_representation_index);
                    segment_size_for_quality_choice=sigma_vals_for_this_user_latest_segment(selected_representation_index);
                else
                    if m_est(user_index) > sqrt(v_est(user_index))
                        U_E_der=1./(m_est(user_index) - sqrt(v_est(user_index) + delta_square_root_approx)+delta_square_root_approx);
                    else
                        U_E_der=1/delta_square_root_approx;
                    end
                    U_E_der=1;
                    U_V_der=1;1./ (2 *( sqrt(v_est(user_index) + delta_square_root_approx)));
                    
                    var_imp_scaling=variability_importance_constant*min(1,(total_number_of_segments_downloaded_by_users(user_index))^2/60^2);
                    
                    fn_b=(1/2)*max(b_est(user_index),0)+(1/2)*max(b_est(user_index)-b_init(user_index)/2,0)+100*max(b_est(user_index)-b_init(user_index),0);
                    %fn_b=max(b_est(user_index)-fn_b_offset(user_index),0);
                    fn_b=fn_b*fn_b_scale(user_index);
                    
                    fn_d=max(d_est(user_index),0);
                    fn_d=fn_d*fn_d_scale(user_index);
                    
                    if use_continuous_curves_approximation==0 || (var_imp_scaling * U_V_der)<0.001                      
                        tmp_metric=U_E_der*( q_vals_for_this_user_latest_segment ...
                            - var_imp_scaling * U_V_der.*( ( q_vals_for_this_user_latest_segment - m_est(user_index)).^2 ) )...
                            - (fn_b/((1+beta_bar(user_index))*segment_duration)).*sigma_vals_for_this_user_latest_segment;
                            %- (xi*(fn_d/segment_duration)/ pbar(user_index)).*sigma_vals_for_this_user_latest_segment;
                        
                        [tmp selected_representation_index]=max(tmp_metric);
                        
                        quality_choice_for_this_user=q_vals_for_this_user_latest_segment(selected_representation_index);
                        segment_size_for_quality_choice=sigma_vals_for_this_user_latest_segment(selected_representation_index);                        
                    else%used if use_continuous_curves_approximation==1
                        q_vals_for_this_user_latest_segment=[0 q_vals_for_this_user_latest_segment 70];
                        sigma_vals_for_this_user_latest_segment=[0 sigma_vals_for_this_user_latest_segment 10000];%assuming that rate of 5000 would realize quality fo 70
                        
                        tmp_metric=U_E_der*( q_vals_for_this_user_latest_segment ...
                            - var_imp_scaling * U_V_der.*( ( q_vals_for_this_user_latest_segment - m_est(user_index)).^2 ) )...
                            - (fn_b/((1+beta_bar(user_index))*segment_duration)).*sigma_vals_for_this_user_latest_segment...
                            - (xi*(fn_d/segment_duration)/ pbar(user_index)).*sigma_vals_for_this_user_latest_segment;
                        
                        [tmp selected_representation_index]=max(tmp_metric);
                        tmp_metric2=tmp_metric;
                        tmp_metric2(selected_representation_index)=-inf;
                        [tmp2 selected_representation_index2]=max(tmp_metric2);
                        
                        sigma_slope=(sigma_vals_for_this_user_latest_segment(selected_representation_index)...
                            - sigma_vals_for_this_user_latest_segment(selected_representation_index2))/...
                            (q_vals_for_this_user_latest_segment(selected_representation_index)...
                            - q_vals_for_this_user_latest_segment(selected_representation_index2));
                        
                        quality_choice_for_this_user= m_est(user_index) + ( 1-sigma_slope*(fn_b/((1+beta_bar(user_index))*segment_duration))...
                            -sigma_slope*(xi*(fn_d/segment_duration)/ pbar(user_index)) )/(2*var_imp_scaling * U_V_der);
                        
                        segment_size_for_quality_choice=sigma_vals_for_this_user_latest_segment(selected_representation_index)...
                            +(quality_choice_for_this_user- sigma_vals_for_this_user_latest_segment(selected_representation_index))/(sigma_slope);
                        
                        if quality_choice_for_this_user>q_vals_for_this_user_latest_segment(selected_representation_index)
                            quality_choice_for_this_user=q_vals_for_this_user_latest_segment(selected_representation_index);
                            segment_size_for_quality_choice=sigma_vals_for_this_user_latest_segment(selected_representation_index);
                        elseif quality_choice_for_this_user<q_vals_for_this_user_latest_segment(selected_representation_index2)
                            quality_choice_for_this_user=q_vals_for_this_user_latest_segment(selected_representation_index2);
                            segment_size_for_quality_choice=sigma_vals_for_this_user_latest_segment(selected_representation_index2);
                        end
                        
                    end
                    if user_index==11 && total_number_of_segments_downloaded_by_users(user_index)>=93
                        'bkpt';
                    end
                    'bkpt';
                end                
            elseif quality_adaptation_scheme==1%greedy segment selection based on current estimate of throughput
                selected_representation_index=max(1,sum((sigma_vals_for_this_user_latest_segment<=((1-.01)*rho_est(user_index)))));
                quality_choice_for_this_user=q_vals_for_this_user_latest_segment(selected_representation_index);
                segment_size_for_quality_choice=sigma_vals_for_this_user_latest_segment(selected_representation_index);                
            elseif quality_adaptation_scheme==2%price control with greedy segment selection based on current estimate of throughput
                selected_representation_index=min(max(1,sum((sigma_vals_for_this_user_latest_segment<=((1-.01)*rho_est(user_index))))),...
                    sum((sigma_vals_for_this_user_latest_segment<= (pbar(user_index)*segment_duration/xi)  )));
                quality_choice_for_this_user=q_vals_for_this_user_latest_segment(selected_representation_index);
                segment_size_for_quality_choice=sigma_vals_for_this_user_latest_segment(selected_representation_index);                
            end
        end
        
        
        
        epsilon_seg=epsilon;max(epsilon,1/(total_number_of_segments_downloaded_by_users(user_index)));        
        m_est(user_index)=m_est(user_index) + epsilon_seg*( quality_choice_for_this_user- m_est(user_index) );
        v_est(user_index)=v_est(user_index) + epsilon_seg*( (quality_choice_for_this_user - m_est(user_index))^2 - v_est(user_index)  );
        sigma_est(user_index) = sigma_est(user_index) + epsilon_seg*(segment_size_for_quality_choice -  sigma_est(user_index));
        d_est(user_index) =  d_est(user_index) + epsilon_seg * ( (xi*segment_size_for_quality_choice/segment_duration)/ pbar(user_index) - 1)  ;
        
        %b_est(user_index)=b_est(user_index)+.0001*(  (sigma_est(user_index)./((1+beta_bar(user_index))*segment_duration))  - rho_est(user_index));                
        %b_est(user_index)=b_est(user_index)+.01*sign(  (sigma_est(user_index)./((1+beta_bar(user_index))*segment_duration))  - rho_est(user_index));      
        
        
        difference=(  (sigma_est(user_index)./((1+beta_bar(user_index))*segment_duration))  - rho_est(user_index));
        %difference=(  (segment_size_for_quality_choice./((1+beta_bar(user_index))*segment_duration))  - rho_est(user_index));
        adiff=abs(difference);sdiff=sign(difference);cur_b=b_est(user_index);        
        
        if cur_b<0.05
            correction=sdiff*min(.0001*adiff,.0005);
        else
            correction=sdiff*min(.0001*adiff,.01);
        end
        
        b_est(user_index)=b_est(user_index)-1;
        
        %b_est(11)=2;
        
        %b_est(user_index)=.01*sdiff*(adiff^2/500^2);
        %b_est(user_index)=max(b_est(user_index),0);
        
        downloaded_segments_quality_data_matrix(total_number_of_segments_downloaded_by_users(user_index)+1,user_index)=...
            quality_choice_for_this_user;
        downloaded_segments_size_data_matrix(total_number_of_segments_downloaded_by_users(user_index)+1,user_index)=...
            segment_size_for_quality_choice;
        
        size_of_current_segment_being_downloaded(user_index)=segment_size_for_quality_choice;
        quality_of_current_segment_being_downloaded(user_index)=quality_choice_for_this_user;
        
        pending_amount_of_data_for_current_segment(user_index)=pending_amount_of_data_for_current_segment(user_index)+segment_size_for_quality_choice;
        set_of_users_who_finish_segment_transmission_in_current_slot=find(pending_amount_of_data_for_current_segment<0);
    end
    
    count_rebuff_slots=count_rebuff_slots+ (( ( (slot-count_rebuff_slots)*slot_duration/segment_duration)- total_number_of_segments_downloaded_by_users-3)>0);
    
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
end

%Evaluate quality metrics
mean_quality=zeros(1,num_users);
quality_std=zeros(1,num_users);
quality_change=zeros(1,num_users);
for user_index=1:num_users
    
    if user_index==11
        'bkpt'
    end
    load('Chao_model');
    tmp= downloaded_segments_quality_data_matrix(1:min_num_of_segments_to_simulate_for_each_user,user_index );
    tmpy=zeros(size(tmp));
    tmpx=chao_sigmoid(tmp-50,w(1),w(2),w(3),w(4));
    for t=1:min_num_of_segments_to_simulate_for_each_user
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
    
    mean_quality(user_index)=mean( downloaded_segments_quality_data_matrix(1:min_num_of_segments_to_simulate_for_each_user,user_index ));
    quality_std(user_index)=std( downloaded_segments_quality_data_matrix(1:min_num_of_segments_to_simulate_for_each_user,user_index ));
    quality_change(user_index)=sqrt(mean(diff( downloaded_segments_quality_data_matrix(1:min_num_of_segments_to_simulate_for_each_user,user_index )).^2));
    price_per_sec(user_index)=xi*mean( downloaded_segments_size_data_matrix(1:min_num_of_segments_to_simulate_for_each_user,user_index ))/segment_duration;
end
QoE=mean_quality -  quality_std;
rebuf_time_est=(slot_duration*slot)- total_number_of_segments_downloaded_by_users*segment_duration;
tmp=[chao_metric;QoE;mean_quality-quality_change;mean_quality;quality_std;quality_change;rebuf_time_est;count_rebuff_slots*slot_duration;price_per_sec;total_allocated_rate/slot];
performance_metrics=[mean(tmp,2) (max(tmp')-min(tmp'))'./mean(tmp,2) tmp ]

write_data=[ [fnarg(1:5)'; variability_importance_constant;min_num_of_segments_to_simulate_for_each_user;mean(beta_bar);fnarg(6);ch_gain_scale] performance_metrics];
dlmwrite(strcat('results',num2str(fnarg(7)),'.txt'),write_data,'-append')
dlmwrite(strcat('results',num2str(fnarg(7)),'.txt'),repmat(123123123,1,size(write_data,2)),'-append')


if 1 % to see traces of users' performance
    if traces_needed==1
        get_traces_for_users=[11];
        store_data_index=[1 2 0 0 0 3 4 0 0 0 5 6 0 0 0];
        for t_user_index=[get_traces_for_users]
            assert(store_data_index(t_user_index)>0);
            performance_metrics(:,t_user_index)
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
beep
returnstatus_INDEX=1;



function rate_allocation_RVBAR=RVBAR(ch_gain_cur_slot)
global num_users eta_RVBAR rmin_RVBAR b_est sigma_est epsilon fn_b_scale fn_b_offset user_present

if sum(rmin_RVBAR./ch_gain_cur_slot)>1
    ch_gain_cur_slot
    input('Even minimum allocation is infeasible; Possible fixes: Reduce min allocation, or Increase average channel quality')
end

fn_b=fn_b_scale.*max(b_est-fn_b_offset,0).*user_present;

tmp=((fn_b.*ch_gain_cur_slot')==max(fn_b.*ch_gain_cur_slot'));
tmp=tmp*(1-(sum( user_present'.*(rmin_RVBAR./ch_gain_cur_slot) )))/sum(tmp);
rate_allocation_RVBAR=ch_gain_cur_slot.*tmp'+rmin_RVBAR*user_present';
assert(abs(sum(rate_allocation_RVBAR./ch_gain_cur_slot)-1) <.0001)

% cvx_begin
% cvx_quiet(true);
% variable rate_allocation_RVBAR(num_users);
% maximize( sum(BS_b_est'.*rate_allocation_RVBAR + eta_RVBAR *sum(sqrt(rate_allocation_RVBAR))) );
% subject to
% sum(rate_allocation_RVBAR./ch_gain_cur_slot)<=1;
% rate_allocation_RVBAR >= rmin_RVBAR*ones(num_users,1);
% cvx_end




function rate_allocation_PF=PF_rate_allocation(ch_gain_cur_slot)
global num_users rmin_RVBAR
check_feasibility_of_closed_form_solution=ch_gain_cur_slot'/num_users;%this is the closed form solution
if min(check_feasibility_of_closed_form_solution)>=rmin_RVBAR
    rate_allocation_PF=check_feasibility_of_closed_form_solution;
else
    if sum(rmin_RVBAR./ch_gain_cur_slot)>1
        ch_gain_cur_slot
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

function rate_allocation_PFL=PF_long_run_rate_allocation(ch_gain_cur_slot)
global rho_est rmin_RVBAR 

if sum(rmin_RVBAR./ch_gain_cur_slot)>1
    ch_gain_cur_slot
    input('Even minimum allocation is infeasible; Possible fixes: Reduce min allocation, or Increase average channel quality')
end

allocation_metric=  (ch_gain_cur_slot'./(rho_est))  ;
tmp=(allocation_metric==max(allocation_metric));

tmp=tmp*(1-(sum(rmin_RVBAR./ch_gain_cur_slot)))/sum(tmp);
rate_allocation_PFL=ch_gain_cur_slot.*tmp'+rmin_RVBAR;
rate_allocation_PFL=rate_allocation_PFL';
assert(abs(sum(rate_allocation_PFL'./ch_gain_cur_slot)-1) <.0001)


function rate_allocation_S=sarabjot_rate_allocation(ch_gain_cur_slot)
global num_users rho_est rmin_RVBAR count_rebuff_slots slot total_number_of_segments_downloaded_by_users segment_duration slot_duration size_of_current_segment_being_downloaded

if sum(rmin_RVBAR./ch_gain_cur_slot)>1
    ch_gain_cur_slot
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
f=30*max(total_number_of_segments_downloaded_by_users*segment_duration-(slot_duration*slot),0); %multiplied by 30 assuming frame rate of 30 fps
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
        ch_gain_cur_slot
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



% function quality_QVBAR=QVBAR()
% global num_users eta_RVBAR rmin_RVBAR m_est v_est b_est d_est sigma_est epsilon
%
%
%
% quality_QVBAR=1;






