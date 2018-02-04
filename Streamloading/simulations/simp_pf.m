%important_constant_1: importance of variability;
%discretization will cause following problems:
%if too large will not allow increase in quality even when no rebuffering issues

%important_constant_2
%decides the sensitivity of corrections in b_est

function return_status_INDEX=simp_pf(fnarg)

%   determining the global variables. For now, we will be using the
%simple linear function instead of the introduced lipschitz function.
global base_size_index num_users rmin_RVBAR m_est v_est b_base_est b_enh_est d_est rho_est sigma_est epsilon  user_present...
    count_rebuff_slots beta_enh_bar beta_base_bar slot total_enh t_seg t_slot size_of_current_segment_being_downloaded...
    total_base svc_overhead b_base_init b_enh_init fn_b_scale enh_layer fn_b_offset

%Take the input vector fnarg and assign values for respective variables.
%close all;clc;
min_seg_sim=1200;num_users=fnarg(1);
rate_allocation_scheme=fnarg(2);quality_adaptation_scheme=fnarg(3);
homogeneous_channels=fnarg(4);markov_channels=fnarg(5);
pbar_val=fnarg(6);max_base_segments=fnarg(8);

t_seg=1;%in seconds
t_slot=0.01;%in seconds
svc_overhead = fnarg(9);

%initial_num_segments_to_be_downloaded_at_low_quality=0; %%   What is this for? (In order to fill up the buffer to some extent?)
%   the below line uses an initial proportional fairness before starting
%to run the actual algortihm.
LPF_run_slots_before_using_RVBAR=0;%needed to ensure that there is no rebuffering in the beginning for the weakest users due to poor allocation
segments_to_run_greedy_before_QVBAR=0;%   what for ?
variability_importance_constant=.1;%important_constant_1.    is that the same eta?

count_rebuff_slots = zeros(1,num_users);
user_present = ones(1,num_users);
beta_enh_bar = 0 * ones(1,num_users);
beta_base_bar = -0.2 * ones(1,num_users);

%The base_size_index determines the size of the base layer with
%respect to the whole segment size. In this regard, it indexes the
%respective encoding size among the 6 available compression rates.

base_size_index = 1;
enh_layer = 6;
epsilon=0.05;
rmin_RVBAR=.001;
xi=.01;

b_base_init=15*epsilon*ones(1,num_users);
b_enh_init=15*epsilon*ones(1,num_users);
fn_b_scale=0.2*ones(1,num_users);
fn_b_offset=0*ones(1,num_users);

%   This is an indicator of the number of outlying base layer segments
%for each user which is incremented periodically and decremented after a
%base layer segment download completion. The assumption is that the
%download begins with a full base layer buffer, where the last allowable base layer segment finished
%downloading at t=0. Hence, at t=0, the next base layer segment becomes
%available.

d_init = ones(1,num_users);
total_allocated_rate=0*ones(1,num_users);

%   I suppose the 2 below lines are just an initialization.
size_of_current_segment_being_downloaded=100*ones(1,num_users);
quality_of_current_segment_being_downloaded=20*ones(1,num_users);


%parameter_initialization
m_est=25*ones(1,num_users);
v_est=5*ones(1,num_users);
b_base_est=b_base_init;
b_enh_est=b_enh_init;
d_est=d_init;

%   what is sgima?
sigma_est=100*ones(1,num_users);
%   Rho is the average throughput.
rho_est=100*ones(1,num_users);


tmp='generate_quality_function_data/data/quality_QR_data_N=';
tmp=strcat(tmp,num2str(33),'_T=',num2str(4500));
quality_QR_data=dlmread(tmp);
quality_QR_data=100-quality_QR_data;
quality_QR_data=reshape(quality_QR_data,33,4500,6);
quality_QR_data=quality_QR_data([1:(0+(num_users/3)) 12:(11+(num_users/3)) 23:(22+(num_users/3))],:,:);
quality_QR_data = repmat(quality_QR_data,1,3);

tmp='generate_quality_function_data/data/rate_QR_data_N=';
tmp=strcat(tmp,num2str(33),'_T=',num2str(4500));
rate_QR_data=dlmread(tmp);
rate_QR_data=reshape(rate_QR_data,33,4500,6);
rate_QR_data=rate_QR_data([1:(0+(num_users/3)) 12:(11+(num_users/3)) 23:(22+(num_users/3))],:,:);
rate_QR_data = repmat(rate_QR_data,1,3);


%tmp='generate_channel_data_code/data/new1_chdata_N=';
tmp='generate_channel_data_code/data/corr_5/chdata_N=';
tmp=strcat(tmp,num2str(33),'_T=',num2str(150000),'_round=',num2str(fnarg(10)));
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
ch_gain=ch_gain([1:(0+(num_users/3)) 12:(11+(num_users/3)) 23:(22+(num_users/3))], : ) ;
ch_gain = repmat(ch_gain,1,3);

tot_qual_matrix=zeros(3*min_seg_sim,num_users);
seg_size=zeros(min_seg_sim+1,num_users);

%trace_info = zeros(14,10000*30);
%intitalization
slot = 0;%index for the current slot
total_enh = -1*ones(1,num_users);

%starting with a full base layer buffer or an empty one
base_size_data_matrix = rate_QR_data(:,:,base_size_index);

%total_base=max_base_segments * ones(1,num_users);
%pending_base = zeros(1,num_users);
total_base = zeros(1,num_users);
pending_base = (1 + svc_overhead)*base_size_data_matrix(:,1)';

tot_qual_matrix(1:min_seg_sim,:) = quality_QR_data(1:num_users,1:min_seg_sim,base_size_index)';
pending_enh = seg_size(1,:);
qual_sel = zeros(min_seg_sim,num_users);
qual_sel_q = zeros(min_seg_sim,num_users);

%Below vectors keep track of late arrived enhancement layers
attempted_enh = zeros(3*4500,num_users,2);

%zeros refers to base mode and 1 refers to enhancement mode
user_mode = zeros(1,num_users);
temp_mode = zeros(1,num_users);

while min(total_base) <= min_seg_sim
    
    slot = slot + 1;
    
    %rate allocation code goes here
    if rate_allocation_scheme == -1
        rate_alloc = 10 * rand(1,num_users); %scheme for testing code
    elseif rate_allocation_scheme == 0 %RVBAR
        if slot < LPF_run_slots_before_using_RVBAR %Use proportional fairness.
            rate_alloc = PF_long_run_rate_allocation(ch_gain(:,slot));
        else
            rate_alloc = RVBAR(ch_gain(:,slot));
        end
        epsilon_slot = max(.001,1/slot);
    elseif rate_allocation_scheme == 3%PF scheme (long time average fairness)
        rate_alloc = PF_long_run_rate_allocation(ch_gain(:,slot));
        epsilon_slot = .1;
    end
    
    
    total_allocated_rate = total_allocated_rate + rate_alloc;
    rho_est = rho_est + epsilon_slot * ((rate_alloc/t_slot) - rho_est);
    b_base_est = b_base_est + epsilon * t_slot ./ (t_seg .* (1 + beta_base_bar));
    b_enh_est = b_enh_est + epsilon * t_slot ./ (t_seg .* (1 + beta_enh_bar));
    
    pending_base(user_mode == 0) = pending_base(user_mode == 0) - rate_alloc(user_mode == 0);
    pending_enh(user_mode == 1) = pending_enh(user_mode == 1) - rate_alloc(user_mode == 1);
    
    
    %Deciding about the base layers delivered.
    base_check = find(pending_base <= 0 & user_mode == 0);
    if(~isempty(base_check))
        total_base(base_check) = total_base(base_check) + 1;
        b_base_est(base_check) = max(b_base_est(base_check) - epsilon*t_seg,0);
        pending_base(base_check) = pending_base(base_check)+...
                (1+svc_overhead) * diag(base_size_data_matrix(pending_base <= 0 & user_mode == 0,...
                total_base(base_check)))';
    end
    temp_mode((total_base * t_seg) - ((slot - count_rebuff_slots) * t_slot) >= max_base_segments) = 1;

    %Determine enhancement layer late arrivals
   front_enh = find((slot - count_rebuff_slots) * t_slot/t_seg - total_enh > 0 & user_mode == 1);
   if ~isempty(front_enh) && slot > 3 * t_seg/t_slot
       for u = front_enh
          if total_enh(u) ~= -1
               if attempted_enh(total_enh(u) + 1,u,1) == 1
                   attempted_enh(total_enh(u) + 1,u,:) = [-1,slot];
               end
           end
           tot_qual_matrix((max(1,total_enh(u) + 1)):ceil((slot - count_rebuff_slots(u)) * t_slot/t_seg),u)=...
               quality_QR_data(u,(max(1,total_enh(u) + 1)):ceil((slot - count_rebuff_slots(u)) * t_slot/t_seg),base_size_index)';
           total_enh(u) = ceil((slot - count_rebuff_slots(u)) * t_slot/t_seg) - 1;
           b_enh_est(u) = max(b_enh_est(u) - epsilon * (ceil((slot - count_rebuff_slots(u)) * t_slot/t_seg) - total_enh(u)),0);
           pending_enh(u) = 0 - rate_alloc(u);
       end
       pending_enh(setdiff(1:num_users,front_enh)) = pending_enh(setdiff(1:num_users,front_enh)) ...
           - rate_alloc(setdiff(1:num_users,front_enh));
   else
       pending_enh(user_mode == 1) = pending_enh(user_mode == 1) - rate_alloc(user_mode == 1);
   end
    
    finish_enh = find(pending_enh <= 0);
     
    %This whole loop is only for those users who have
    %completely downloaded their latest segment.
    for user_index = finish_enh
        
        flag = false;
        if quality_adaptation_scheme==0 %QVBAR
            while (flag == false)
                total_enh(user_index) = total_enh(user_index) + 1;
                qual_val = reshape(quality_QR_data(user_index,total_enh(user_index) + 1,:),1,enh_layer);
                last_sigma = (1 + svc_overhead) * reshape(rate_QR_data(user_index,total_enh(user_index)+1,:),1,enh_layer);
                sigma_enh = last_sigma - last_sigma(base_size_index);
                
                
                fn_b_enh = fn_b_scale(user_index) * ((1/2) * max(b_enh_est(user_index),0) + (1/2) * max(b_enh_est(user_index) - b_enh_init(user_index)/2,0)...
                    + 100 * max(b_enh_est(user_index) - b_enh_init(user_index),0));
                
                tmp_metric = qual_val - fn_b_enh/((1 + beta_enh_bar(user_index)) * t_seg).*sigma_enh ...
                    - variability_importance_constant * ( qual_val - m_est(user_index)).^2;
                
                
                [~, selected_representation_index] = max(tmp_metric);
                selected_representation_index = max(selected_representation_index,base_size_index);            
                user_quality = qual_val(selected_representation_index);
                enh_size_for_quality_choice = sigma_enh(selected_representation_index);
                qual_sel(total_enh(user_index) + 1,user_index) = selected_representation_index;
                qual_sel_q(total_enh(user_index) + 1,user_index) = user_quality;
                
                if selected_representation_index > base_size_index
                    flag = true;
                else
                    b_enh_est(user_index) = max(b_enh_est(user_index) - epsilon * t_seg,0);
                end
                
                m_est(user_index) = m_est(user_index) + epsilon * ( user_quality - m_est(user_index) );
                tot_qual_matrix(total_enh(user_index) + 1,user_index) = user_quality;
                seg_size(total_enh(user_index) + 1,user_index) = ...
                    enh_size_for_quality_choice + base_size_data_matrix(user_index,total_enh(user_index)+1);
                size_of_current_segment_being_downloaded(user_index) = enh_size_for_quality_choice;
                quality_of_current_segment_being_downloaded(user_index) = user_quality;
                pending_enh(user_index) = pending_enh(user_index) + enh_size_for_quality_choice;
                
                if((total_base(user_index) * t_seg) - ((slot - count_rebuff_slots) * t_slot) < max_base_segments)
                    user_mode(user_index) = 0;
                end
            end
        end
        if quality_adaptation_scheme == 1 %&& rm_counter(user_index) <= 0%greedy segment selection based on current estimate of throughput
            total_enh(user_index) = total_enh(user_index) + 1;
            qual_val = reshape(quality_QR_data(user_index,total_enh(user_index) + 1,:),1,enh_layer);
            last_sigma = (1 + svc_overhead) * reshape(rate_QR_data(user_index,total_enh(user_index)+1,:),1,enh_layer);
            sigma_enh = last_sigma - last_sigma(base_size_index);
            
            selected_representation_index = max(1,sum((sigma_enh<=((1 - .01)*rho_est(user_index)))));
            selected_representation_index = max(selected_representation_index,base_size_index);
            
            user_quality=qual_val(selected_representation_index);
            enh_size_for_quality_choice=sigma_enh(selected_representation_index);
            
            tot_qual_matrix(total_enh(user_index) + 1,user_index) = user_quality;
            m_est(user_index) = m_est(user_index) + epsilon * (user_quality - m_est(user_index));
            seg_size(total_enh(user_index) + 1,user_index)=...
                enh_size_for_quality_choice + base_size_data_matrix(user_index,total_enh(user_index) + 1);
            size_of_current_segment_being_downloaded(user_index) = enh_size_for_quality_choice;
            quality_of_current_segment_being_downloaded(user_index) = user_quality;
            pending_enh(user_index) = pending_enh(user_index) + enh_size_for_quality_choice;
            if((total_base(user_index) * t_seg) - ((slot - count_rebuff_slots) * t_slot) < max_base_segments)
                    user_mode(user_index) = 0;
            end
        end
        
        %The rest of the parameters are updated here in case of
        %segment completetion.
        epsilon_seg = epsilon;%max(epsilon,1/(total_number_of_segments_downloaded_by_users(user_index)));
        v_est(user_index) = v_est(user_index) + epsilon_seg * ((user_quality - m_est(user_index))^2 - v_est(user_index));
        sigma_est(user_index) = sigma_est(user_index) + epsilon_seg * (enh_size_for_quality_choice -  sigma_est(user_index));
        
        b_enh_est(user_index) = max(b_enh_est(user_index) - epsilon * t_seg,0);
    end
    user_mode(temp_mode == 1) = 1;
    temp_mode = zeros(num_users,1);
    
    count_rebuff_slots = count_rebuff_slots + ((((slot-count_rebuff_slots) * t_slot/t_seg)- total_base) - 3 > 0);
end

mean_quality=zeros(1,num_users);
quality_std=zeros(1,num_users);
quality_change=zeros(1,num_users);
out_of_time_enh = zeros(1,num_users);
for user_index = 1:num_users
    
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
    out_of_time_enh(user_index) = sum(attempted_enh(1:min_seg_sim,user_index,1) == -1);
end

QoE = mean_quality - quality_std;
rebuf_time_est = (t_slot * slot) - total_enh * t_seg;
tmp=[chao_metric;QoE;mean_quality-quality_change;mean_quality;quality_std;quality_change;rebuf_time_est;count_rebuff_slots*t_slot;out_of_time_enh;total_allocated_rate/slot];
performance_metrics=[mean(tmp,2) (max(tmp')-min(tmp'))'./mean(tmp,2) tmp ];

write_data=[ [fnarg(1:5)'; variability_importance_constant;min_seg_sim;mean(beta_enh_bar);fnarg(6);ch_gain_scale] performance_metrics];
dlmwrite(strcat('results',num2str(fnarg(7)),'.txt'),write_data,'-append')
dlmwrite(strcat('results',num2str(fnarg(7)),'.txt'),repmat(123123123,1,size(write_data,2)),'-append')

returnstatus_INDEX=1;

function rate_allocation_RVBAR=RVBAR(ch_gain_cur_slot)
global num_users rmin_RVBAR b_enh_est b_base_est user_present fn_b_scale fn_b_offset
rate_allocation_RVBAR = zeros(1,num_users);
if ~isempty(base_dl)
    
    fn_b=fn_b_scale(base_dl).*max(b_base_est(base_dl)-...
        fn_b_offset(base_dl),0).*user_present(base_dl);
    ch_gain_cur_slot = ch_gain_cur_slot(base_dl);
    users_active = user_present(base_dl);
    
else
    fn_b=fn_b_scale.*max(b_enh_est-...
        fn_b_offset,0).*user_present;
    users_active = user_present;
end

if sum(rmin_RVBAR./ch_gain_cur_slot)>1
    input('Even minimum allocation is infeasible; Possible fixes: Reduce min allocation, or Increase average channel quality')
end

tmp=((fn_b.*ch_gain_cur_slot')==max(fn_b.*ch_gain_cur_slot'));
tmp=tmp*(1-(sum( users_active'.*(rmin_RVBAR./ch_gain_cur_slot) )))/sum(tmp);
rate_allocation_RVBAR_tmp=ch_gain_cur_slot.*tmp'+rmin_RVBAR*users_active';
rate_allocation_RVBAR_tmp=rate_allocation_RVBAR_tmp';
if ~isempty(base_dl)
    rate_allocation_RVBAR(base_dl) = rate_allocation_RVBAR_tmp;
else
    rate_allocation_RVBAR = rate_allocation_RVBAR_tmp;
end
%assert(abs(sum(rate_allocation_RVBAR_tmp'./ch_gain_cur_slot)-1) <.0001;
      
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

function y=chao_sigmoid(x,a,b,c,d)
y=d./(1+exp(-(a*x+b)))+c;

