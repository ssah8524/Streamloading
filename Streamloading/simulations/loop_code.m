clc
%close all

generate_new_data=0; % set to 1 to generate new channel and QR curve data

num_user_vector=[3 6 9 12 15 18 21 24 27];
%num_user_vector=[27];

tic
for simulation_cnt=1:10
    simulation_cnt
    
    if generate_new_data==1    
        cd generate_channel_data_code
        gen_channel_data_mine(33);
        cd ..
    end
        cd generate_quality_function_data
        gen_quality_fn_data(33);
        cd ..
    
    
    for pbar=20000%[20000 2 4 6]        
        %for num_users=[15 ]
        %for rate_allocation_scheme=[3]
            for quality_adaptation_scheme=[0]
                for num_users=num_user_vector
                    
                    %if rate_allocation_scheme==0 && quality_adaptation_scheme~=0
                        
                    %else
                        num_users
                                                
                        %runsim([num_users,0,quality_adaptation_scheme,1,1,pbar,100,simulation_cnt]);
                        %runsim([num_users,3,quality_adaptation_scheme,1,1,pbar,100,simulation_cnt]);

                        %runsim_orig_test([num_users,rate_allocation_scheme,quality_adaptation_scheme,1,1,pbar,10,50,simulation_cnt]);
                        %runsim_sl([num_users,0,quality_adaptation_scheme,1,1,pbar,1002,50,0.1,simulation_cnt]);
                        %runsim_sl_lim([num_users,0,quality_adaptation_scheme,1,1,pbar,103,50,0.1,simulation_cnt,50]);
                        %runsim_sl_lim([num_users,0,quality_adaptation_scheme,1,1,pbar,102,50,0.1,simulation_cnt,150]);
                        %runsim_sl_lim([num_users,0,quality_adaptation_scheme,1,1,pbar,102,50,0.1,simulation_cnt,200]);

                        %simp_pf([num_users,3,1,1,1,pbar,1,50,0.1,simulation_cnt]);

                    %end
                end
            end
        %end
    end
end
toc
