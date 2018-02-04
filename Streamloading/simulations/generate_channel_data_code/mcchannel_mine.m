% This function can be used to generate a channel capacity for the next 
% time slot given the current channel capacity according to a Markov Chain.
% The Markov chain is created by the Metropolis Algorithm using the AT&T data.
% c1: current channel; c2: the next channel
% TPD: The AT&T channel data matrix (the throughput distribution). Use 
% "load TPD" to load it.
% rtype: receive type (2: single antenna,3: dual antenna)
% ltype: load level (1:25%,2:50%,3:100%).Ususally we can use rtype=2;ltype=2;
% granu: granuality. Since the channel is discretized in the markov chain,
% granu is the granuality of the quatization. e.g., granu=1 means we only
% use the capacity in the matrix TPD 250, 500, 750,1000... granu=2
% corresponds to 125, 250, 375, 500, ...
% range: the range of the neighbors of each state in the Markov chain. 
% e.g., range=1 means we only allow the next state to be the left one, 
% the right one or stay in the same state. Note: N(x)= 2*range+1 

function [c2,out_flag] = mcchannel_mine(c1,ChannelPDF,range)

% get transition probabilities
k=find(ChannelPDF(1,:)==c1);
M=2*range+1;
transprob=zeros(M+1,1);
j=1;
for i=k-range:k+range
    if i==k
        indcur=j;
        j=j+1;
    end
    if i>0 && i<=length(ChannelPDF(1,:)) && i~=k
        transprob(j)=1/M*min(1,ChannelPDF(2,i)/ChannelPDF(2,k));
        j=j+1;
    end
end
transprob(indcur)=1-sum(transprob); %probability of staying in the same state.
n=find(transprob==0,1);
transprob=transprob(1:n-1);
transcdf=zeros(n-1,1);
for i=1:n-1
    transcdf(i)=sum(transprob(1:i));
end

% get the next channel
% get the next channel
r=rand;
if r==1
    c2=ChannelPDF(1,k+n-1-indcur);
else
    i=find(transcdf>r,1);
    c2=ChannelPDF(1,k+i-indcur);
end