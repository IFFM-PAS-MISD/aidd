
fs=100000*32;dt=1/fs;nft=2^12;f_1=20000;f_2=100000;t_1=0;[t,st]=Hanning_signal(dt,nft,f_1,f_2,t_1);
f=([1:nft]-1)*fs/nft;
figure; plot(f/1e3,abs(fft(st,nft))/nft);
xlabel('Frequency [kHz]');ylabel('Magnitude');xlim([0 300])
fs=100000*32;dt=1/fs;nft=2^12;f_1=5000;f_2=100000;t_1=0;[t,st]=Hanning_signal(dt,nft,f_1,f_2,t_1);
f=([1:nft]-1)*fs/nft;
figure; plot(f/1e3,abs(fft(st,nft))/nft);
xlabel('Frequency [kHz]');ylabel('Magnitude');xlim([0 300]);