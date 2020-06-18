mkdir -p log_files/transfer

log_file=log_files/transfer/exp_101_pretrained_nturgbd_skelcont_test_ucf101_all_splits.log
log_file_training=log_files/transfer/exp_101_pretrained_nturgbd_skelcont_test_ucf101_all_splits_training.log

network_split_1="/home/dschneider/workspace/DPC/eval/20200616094118_log_pt_nturgbd_cont_transfer_ucf101_sp_1_ep_200_ft/ucf101-128-sp1_r18_lc_cont_bs60_lr0.0001_wd0.001_ds1_seq1_len30_dp0.5_train-ft_pt=pretrained_net/model/model_best_epoch129.pth.tar"
network_split_2="/home/dschneider/workspace/DPC/eval/20200616094118_log_pt_nturgbd_cont_transfer_ucf101_sp_1_ep_200_ft/ucf101-128-sp1_r18_lc_cont_bs60_lr0.0001_wd0.001_ds1_seq1_len30_dp0.5_train-ft_pt=pretrained_net/model/model_best_epoch129.pth.tar"
network_split_3="/home/dschneider/workspace/DPC/eval/20200617065118_log_pt_nturgbd_cont_transfer_ucf101_sp_3_ep_200_ft/ucf101-128-sp3_r18_lc_cont_bs60_lr0.0001_wd0.001_ds1_seq1_len30_dp0.5_train-ft_pt=pretrained_net/model/model_best_epoch147.pth.tar"


echo "Start test of network finetuned on UCF101 train split 1:" > $log_file
date >> $log_file
python3 -u test.py --batch_size 1 --dataset ucf101 --split 1 --seq_len 30 --num_seq 1 --ds 1 --num_workers 16 --gpu 0 --test $network_split_1 2>&1 | tee log_file_training
echo "Finished testing UCF101 split 1 with exit code $?" >> $log_file
date >> $log_file
echo "" >> $log_file

echo "Start test of network finetuned on UCF101 train split 2:" >> $log_file
date >> $log_file
python3 -u test.py --batch_size 1 --dataset ucf101 --split 2 --seq_len 30 --num_seq 1 --ds 1 --num_workers 16 --gpu 0 --test $network_split_2 2>&1 | tee -a log_file_training
echo "Finished testing UCF101 split 2 with exit code $?" >> $log_file
date >> $log_file
echo "" >> $log_file

echo "Start test of network finetuned on UCF101 train split 3:" >> $log_file
date >> $log_file
python3 -u test.py --batch_size 1 --dataset ucf101 --split 3 --seq_len 30 --num_seq 1 --ds 1 --num_workers 16 --gpu 0 --test $network_split_3 2>&1 | tee -a log_file_training
echo "Finished testing UCF101 split 3 with exit code $?" >> $log_file
date >> $log_file
echo "" >> $log_file
