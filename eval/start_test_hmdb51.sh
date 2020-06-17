mkdir -p log_files

log_file=log_files/transfer/exp_100-0_pretrained_nturgbd_test_hmdb51_all_splits.log
log_file_training=log_files/transfer/exp_100-0_temp_test_pretrained_nturgbd_test_hmdb51_all_splits_training.log

network_split_1="/home/dschneider/workspace/DPC/eval/20200614195942_log_pt_nturgbd_cont_transfer_hmdb51_sp_1_ep_200_ft/hmdb51-128-sp1_r18_lc_cont_bs20_lr0.0001_wd0.001_ds1_seq1_len30_dp0.5_train-ft_pt=pretrained_net/model/model_best_epoch153.pth.tar"  # "/home/dschneider/workspace/DPC/eval/log_pt_nturgbd_eval_hmdb51_sp_1_ep_200_ft/hmdb51-128-sp1_r18_lc_bs96_lr0.001_wd0.001_ds1_seq6_len5_dp0.5_train-ft_pt=pretrained_net/model/model_best_epoch158.pth.tar"
network_split_2="/home/dschneider/workspace/DPC/eval/20200616005706_log_run2_pt_nturgbd_cont_transfer_hmdb51_sp_2_ep_200_ft/hmdb51-128-sp2_r18_lc_cont_bs30_lr0.0001_wd0.001_ds1_seq1_len30_dp0.5_train-ft_pt=pretrained_net/model/model_best_epoch55.pth.tar"
network_split_3="/home/dschneider/workspace/DPC/eval/20200616142615_log_run2_pt_nturgbd_cont_transfer_hmdb51_sp_3_ep_200_ft/hmdb51-128-sp3_r18_lc_cont_bs30_lr0.0001_wd0.001_ds1_seq1_len30_dp0.5_train-ft_pt=pretrained_net/model/model_best_epoch155.pth.tar"


echo "Start test of network finetuned on HMDB51 train split 1:" | tee $log_file
date | tee -a $log_file
# python3 test.py --dataset hmdb51 --split 1 --seq_len 30 --num_seq 1 --ds 1 --num_workers 16 --gpu 1 --test $network_split_1 # > $log_file_training
echo "Finished testing HMDB51 split 1 with exit code $?" >> $log_file
date | tee -a $log_file
echo "" | tee -a $log_file

echo "Start test of network finetuned on HMDB51 train split 2:" >> $log_file
date >> $log_file
python3 test.py --dataset hmdb51 --split 2 --seq_len 30 --batch_size 1 --num_seq 1 --ds 1 --num_workers 16 --gpu 0 --test $network_split_2  2>&1 | tee $log_file_training
echo "Finished testing HMDB51 split 2 with exit code $?" >> $log_file
date >> $log_file
echo "" >> $log_file

echo "Start test of network finetuned on HMDB51 train split 3:" >> $log_file
date >> $log_file
python3 test.py --dataset hmdb51 --split 3 --seq_len 30 --num_seq 1 --batch_size 1 --ds 1 --num_workers 16 --gpu 0 --test $network_split_3 2>&1 | tee -a $log_file_training
echo "Finished testing HMDB51 split 3 with exit code $?" >> $log_file
date >> $log_file
echo "" >> $log_file
