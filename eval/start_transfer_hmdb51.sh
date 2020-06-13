mkdir -p log_files

log_file=log_files/pretrained_nturgbd_transfer_hmdb51_all_splits.log
log_file_training=log_files/pretrained_nturgbd_transfer_hmdb51_all_splits_training.log
pretrained_net="../dpc/log_tmp/nturgbd-128_r18_dpc-rnn_bs96_lr0.001_seq6_pred2_len5_ds1_train-all/model/model_best_epoch85.pth.tar"

echo "Start of transfer training on HMDB51 split 2:" > $log_file
date >> $log_file
python3 test.py --prefix pt_nturgbd_eval_hmdb51_sp_2_ep_200_ft --dataset hmdb51 --split 2 --train_what ft --epochs 200 --seq_len 5 --num_seq 6 --ds 1 --batch_size 96 --num_workers 32 --gpu 4 5 --pretrain $pretrained_net > $log_file_training
echo "Finished evaluation training on HMDB51 split 2 with exit code $?" >> $log_file
date >> $log_file

echo "" >> $log_file

echo "Start of evaluation training on HMDB51 split 3:" >> $log_file
date >> $log_file
python3 test.py --prefix pt_nturgbd_eval_hmdb51_sp_3_ep_200_ft --dataset hmdb51 --split 3 --train_what ft --epochs 200 --seq_len 5 --num_seq 6 --ds 1 --batch_size 96 --num_workers 32 --gpu 4 5 --pretrain $pretrained_net >> $log_file_training
echo "Finished evaluation training on HMDB51 split 3 with exit code $?" >> $log_file
date >> $log_file

echo "" >> $log_file

echo "Start of evaluation training on HMDB51 split 1:" >> $log_file
date >> $log_file
python3 test.py --prefix pt_nturgbd_eval_hmdb51_sp_1_ep_200_ft --dataset hmdb51 --split 1 --train_what ft --epochs 200 --seq_len 5 --num_seq 6 --ds 1 --batch_size 96 --num_workers 32 --gpu 4 5 --pretrain $pretrained_net >> $log_file_training
echo "Finished evaluation training on HMDB51 split 1 with exit code $?" >> $log_file
date >> $log_file
