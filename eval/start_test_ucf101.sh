mkdir -p log_files

log_file=log_files/pretrained_nturgbd_test_ucf101_all_splits.log
log_file_training=log_files/pretrained_nturgbd_test_ucf101_all_splits_training.log

network_split_1="/home/dschneider/workspace/DPC/eval/log_log_pt_nturgbd_eval_ucf101_sp_1_ep_200_ft_saturday/ucf101-128-sp1_r18_lc_bs96_lr0.001_wd0.001_ds1_seq6_len5_dp0.5_train-ft_pt=pretrained_net/model/model_best_epoch122.pth.tar"
network_split_2="/home/dschneider/workspace/DPC/eval/log_log_pt_nturgbd_eval_ucf101_sp_2_ep_200_ft_saturday/ucf101-128-sp2_r18_lc_bs96_lr0.001_wd0.001_ds1_seq6_len5_dp0.5_train-ft_pt=pretrained_net/model/model_best_epoch174.pth.tar"
network_split_3="/home/dschneider/workspace/DPC/eval/log_log_pt_nturgbd_eval_ucf101_sp_3_ep_200_ft_saturday/ucf101-128-sp3_r18_lc_bs96_lr0.001_wd0.001_ds1_seq6_len5_dp0.5_train-ft_pt=pretrained_net/model/model_best_epoch162.pth.tar"


echo "Start test of network finetuned on UCF101 train split 1:" > $log_file
date >> $log_file
python3 test.py --dataset ucf101 --split 1 --seq_len 5 --num_seq 6 --ds 1 --num_workers 32 --gpu 4 --test $network_split_1 > log_file_training
echo "Finished testing UCF101 split 1 with exit code $?" >> $log_file
date >> $log_file
echo "" >> $log_file

echo "Start test of network finetuned on UCF101 train split 2:" >> $log_file
date >> $log_file
python3 test.py --dataset ucf101 --split 2 --seq_len 5 --num_seq 6 --ds 1 --num_workers 32 --gpu 4 --test $network_split_2 >> log_file_training
echo "Finished testing UCF101 split 2 with exit code $?" >> $log_file
date >> $log_file
echo "" >> $log_file

echo "Start test of network finetuned on UCF101 train split 3:" >> $log_file
date >> $log_file
python3 test.py --dataset ucf101 --split 3 --seq_len 5 --num_seq 6 --ds 1 --num_workers 32 --gpu 4 --test $network_split_3 >> log_file_training
echo "Finished testing UCF101 split 3 with exit code $?" >> $log_file
date >> $log_file
echo "" >> $log_file
