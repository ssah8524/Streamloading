res=dlmread('results102.txt');
%res=dlmread('results_corr_5/results5000.txt');
num_sim = 10;
[x y] = size(res);
rows = x/11 - 1;
N=res(11*(0:rows)+1,1);
qoe = res(11*(0:rows)+4,2);
qoe_w_change = res(11*(0:rows)+2,2);
rebuf = res(11*(0:rows)+8,2);
out_of_time = res(11*(0:rows) + 9,2);
util = res(11*(0:rows)+10,2);

%fair_qoe=res(11*(0:rows) + 4,3);
qoe = reshape(qoe,length(qoe)/num_sim,num_sim);
qoe_w_change = reshape(qoe_w_change,length(qoe_w_change)/num_sim,num_sim);
rebuf = reshape(rebuf,length(rebuf)/num_sim,num_sim);
out_of_time = reshape(out_of_time,length(out_of_time)/num_sim,num_sim);
util = reshape(util,length(util)/num_sim,num_sim);
%fair_qoe=reshape(fair_qoe,length(fair_qoe)/num_sim,num_sim);

qoe = mean(qoe,2);
qoe_w_change = mean(qoe_w_change,2);
rebuf = mean(rebuf,2);
out_of_time = mean(out_of_time,2);
util = mean(util,2);
%fair_qoe=mean(fair_qoe,2);

figure
hold on
plot([3 6 9 12 15 18 21 24 27],qoe_w_change(1:3:27),'bo-','LineWidth',2)
plot([3 6 9 12 15 18 21 24 27],qoe_w_change(2:3:27),'ro-','LineWidth',2)
plot([3 6 9 12 15 18 21 24 27],qoe_w_change(3:3:27),'go-','LineWidth',2)
hold off


figure
hold on
plot([3 6 9 12 15 18 21 24 27],rebuf(1:3:27),'bo-','LineWidth',2)
plot([3 6 9 12 15 18 21 24 27],rebuf(2:3:27),'ro-','LineWidth',2)
plot([3 6 9 12 15 18 21 24 27],rebuf(3:3:27),'go-','LineWidth',2)
hold off

figure
hold on
plot([3 6 9 12 15 18 21 24 27],rebuf(1:2:18),'bo-','LineWidth',2)
%plot([3 6 9 12 15 18 21 24 27],rebuf(2:2:18)+0.1,'ro-','LineWidth',2)
%plot([3 6 9 12 15 18 21 24 27],rebuf(3:3:27),'go-','LineWidth',2)
hold off


%% For buffer filling analysis
buffer_sl = dlmread('buffer_fill_sl_new.txt');
buffer_st = dlmread('buffer_fill.txt');
buffer_nova = dlmread('buffer_fill_nova.txt');

qual_nova = dlmread('qual_nova.txt');

[x_st y_st] = size(buffer_st);
buffer_sl = dlmread('buffer_fill_sl.txt');
buffer_sl_new = dlmread('buffer_fill_sl_new.txt');

[x_sl y_sl] = size(buffer_sl);
rebuf_st = dlmread('rebuf_dist.txt');
rebuf_sl = dlmread('rebuf_dist_sl.txt');
b_st = dlmread('virtual_buffer.txt');
b_nova = dlmread('virtual_buffer_nova.txt');
b_sl = dlmread('virtual_buffer_sl.txt');
qual_sl = dlmread('quality_sl.txt');
qual_st = dlmread('quality_st.txt');
alloc_sl = dlmread('alloc_sl.txt');
alloc_st = dlmread('alloc_st.txt');


%If NOT used for maximum buffer occupancy, use the following:
%util = res(11*(0:rows)+10,4:36);

% nrow = x/11/num_sim;
% mean_rate = zeros(nrow,33);
% for i = 1:nrow
%     mean_rate(i,:) = mean(util(i:nrow:(x/11),:));
% end
% 
% m = [12 15 18 21 24 27 30 33];
% 
% mean_opt = mean_rate(1:nrow/3,:);
% mean_opt_dl = mean_opt(1:5:size(mean_opt),:); 
% jain_opt_dl = ((sum(mean_opt_dl,2).^2)./diag(mean_opt_dl*mean_opt_dl'))./m';
% mean_opt_sl = mean_opt(1:5:size(mean_opt),:);
% jain_opt_sl = ((sum(mean_opt_sl,2).^2)./diag(mean_opt_sl*mean_opt_sl'))./m';
% 
% mean_pf = mean_rate((nrow/3 + 1):(2*nrow/3),:);
% mean_pf_dl = mean_pf(1:5:size(mean_pf),:);
% jain_pf_dl = ((sum(mean_pf_dl,2).^2)./diag(mean_pf_dl*mean_pf_dl'))./m';
% mean_pf_sl = mean_pf(1:5:size(mean_pf),:);
% jain_pf_sl = ((sum(mean_pf_sl,2).^2)./diag(mean_pf_sl*mean_pf_sl'))./m';
% 
% 
% mean_rm = mean_rate((2*nrow/3 + 1):nrow,:);
% mean_rm_dl = mean_rm(1:5:size(mean_rm),:);
% jain_rm_dl = ((sum(mean_rm_dl,2).^2)./diag(mean_rm_dl*mean_rm_dl'))./m';
% mean_rm_sl = mean_pf(1:5:size(mean_rm),:);
% jain_rm_sl = ((sum(mean_rm_sl,2).^2)./diag(mean_rm_sl*mean_rm_sl'))./m';



qoe_opt = qoe_w_change(1:(length(qoe)/3));
qoe_opt1 = qoe_opt(1:5:length(qoe_opt));
qoe_opt2 = qoe_opt(2:5:length(qoe_opt));
qoe_opt3 = qoe_opt(3:5:length(qoe_opt));

qoe_pf = qoe_w_change((length(qoe)/3 + 1):(2*length(qoe)/3));
qoe_pf1 = qoe_pf(1:5:length(qoe_pf));
qoe_pf2 = qoe_pf(2:5:length(qoe_pf));
qoe_pf3 = qoe_pf(3:5:length(qoe_pf));
qoe_buf = qoe_pf(4:5:length(qoe_pf));
qoe_ov = qoe_pf(5:5:length(qoe_pf));

qoe_pf_rm = qoe_w_change((2*length(qoe)/3 + 1):(3*length(qoe)/3));
qoe_pf_rm1 = qoe_pf_rm(1:5:length(qoe_pf_rm));
qoe_pf_rm2 = qoe_pf_rm(2:5:length(qoe_pf_rm));
qoe_pf_rm3 = qoe_pf_rm(3:5:length(qoe_pf_rm));

rebuf_opt = rebuf(1:(length(rebuf)/3));
rebuf_opt1 = rebuf_opt(1:5:length(rebuf_opt));
rebuf_opt2 = rebuf_opt(2:5:length(rebuf_opt));
rebuf_opt3 = rebuf_opt(3:5:length(rebuf_opt));
rebuf_buf = rebuf_opt(4:5:length(qoe_pf));
rebuf_ov = rebuf_opt(5:5:length(qoe_pf));


rebuf_pf = rebuf((length(rebuf)/3 + 1):(2*length(rebuf)/3));
rebuf_pf1 = rebuf_pf(1:5:length(rebuf_pf));
rebuf_pf2 = rebuf_pf(2:5:length(rebuf_pf));
rebuf_pf3 = rebuf_pf(3:5:length(rebuf_pf));

rebuf_pf_rm = rebuf((2*length(rebuf)/3 + 1):(3*length(rebuf)/3));
rebuf_pf_rm1 = rebuf_pf_rm(1:5:length(rebuf_pf_rm));
rebuf_pf_rm2 = rebuf_pf_rm(2:5:length(rebuf_pf_rm));
rebuf_pf_rm3 = rebuf_pf_rm(3:5:length(rebuf_pf_rm));

fair_opt = fair_qoe(1:(length(rebuf)/3));
fair_opt1 = fair_opt(1:5:length(rebuf_opt));
fair_opt2 = fair_opt(2:5:length(rebuf_opt));
fair_opt3 = fair_opt(3:5:length(rebuf_opt));

fair_pf = fair_qoe((length(rebuf)/3 + 1):(2*length(rebuf)/3));
fair_pf1 = fair_pf(1:5:length(rebuf_pf));
fair_pf2 = fair_pf(2:5:length(rebuf_pf));
fair_pf3 = fair_pf(3:5:length(rebuf_pf));

fair_pf_rm = fair_qoe((2*length(rebuf)/3 + 1):(3*length(rebuf)/3));
fair_pf_rm1 = fair_pf_rm(1:5:length(rebuf_pf_rm));
fair_pf_rm2 = fair_pf_rm(2:5:length(rebuf_pf_rm));
fair_pf_rm3 = fair_pf_rm(3:5:length(rebuf_pf_rm));
 
util_opt = util(1:(length(rebuf)/3));
util_opt1 = util_opt(1:5:length(rebuf_opt));
util_opt2 = util_opt(2:5:length(rebuf_opt));
util_opt3 = util_opt(3:5:length(rebuf_opt));

util_pf = util((length(rebuf)/3 + 1):(2*length(rebuf)/3));
util_pf1 = util_pf(1:5:length(rebuf_pf));
util_pf2 = util_pf(2:5:length(rebuf_pf));
util_pf3 = util_pf(3:5:length(rebuf_pf));

util_pf_rm = util((2*length(rebuf)/3 + 1):(3*length(rebuf)/3));
util_pf_rm1 = util_pf_rm(1:5:length(rebuf_pf_rm));
util_pf_rm2 = util_pf_rm(2:5:length(rebuf_pf_rm));
util_pf_rm3 = util_pf_rm(3:5:length(rebuf_pf_rm));
% qoe sl st dl
figure
plot([12 15 18 21 24 27 30 33],qoe_opt1)
hold on
plot([12 15 18 21 24 27 30 33],qoe_opt2)
hold on
plot([12 15 18 21 24 27 30 33],qoe_opt3)

% rebuf sl st dl
figure
plot([12 15 18 21 24 27 30 33],rebuf_opt1)
hold on
plot([12 15 18 21 24 27 30 33],rebuf_opt2)
hold on
plot([12 15 18 21 24 27 30 33],rebuf_opt3)

% qoe opt pf pf/rm
figure
plot([12 15 18 21 24 27 30 33],qoe_opt3)
hold on
plot([12 15 18 21 24 27 30 33],qoe_pf3)
hold on
plot([12 15 18 21 24 27 30 33],qoe_pf_rm3)

% rebuf opt pf pf/rm
figure
plot([12 15 18 21 24 27 30 33],rebuf_opt3)
hold on
plot([12 15 18 21 24 27 30 33],rebuf_pf3)
hold on
plot([12 15 18 21 24 27 30 33],rebuf_pf_rm3)

%fairness opt pf pf/rm for streamloading
figure
plot([12 15 18 21 24 27 30 33],fair_opt3)
hold on
plot([12 15 18 21 24 27 30 33],fair_pf3)
hold on
plot([12 15 18 21 24 27 30 33],fair_pf_rm3)

%link utilization sl st dl
figure
plot([12 15 18 21 24 27 30 33],100*util_opt1)
hold on
plot([12 15 18 21 24 27 30 33],100*util_opt2)
hold on
plot([12 15 18 21 24 27 30 33],100*util_opt3)

% qoe buffer length comp streamloading
figure
plot([12 15 18 21 24 27 30 33],qoe_opt3)
hold on
plot([12 15 18 21 24 27 30 33],qoe_buf)
hold on
plot([12 15 18 21 24 27 30 33],qoe_opt2)

% rebuf buffer length comp streamloading
figure
plot([12 15 18 21 24 27 30 33],rebuf_opt3)
hold on
plot([12 15 18 21 24 27 30 33],rebuf_buf)
hold on
plot([12 15 18 21 24 27 30 33],rebuf_opt2)

%qoe overhead comparison streamloading
figure
plot([12 15 18 21 24 27 30 33],qoe_opt3)
hold on
plot([12 15 18 21 24 27 30 33],qoe_ov)
hold on
plot([12 15 18 21 24 27 30 33],qoe_opt2)

%rebuf overhead comparison streamloading
figure
plot([12 15 18 21 24 27 30 33],rebuf_opt3)
hold on
plot([12 15 18 21 24 27 30 33],rebuf_ov)
hold on
plot([12 15 18 21 24 27 30 33],rebuf_opt2)

%Jain fairness index
figure
plot(m,jain_opt_sl);
hold on
plot(m,jain_pf_sl);


