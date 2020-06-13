mkdir -p log_files

log_file=log_files/untrained_transfer_hmdb51_all_splits.log
log_file_training=log_files/untrained_transfer_hmdb51_all_splits_training.log

echo "Start training of random network on HMDB51 split 1:" > $log_file
date >> $log_file
python3 test.py --prefix ut_training_hmdb51_sp_1_ep_200 --dataset hmdb51 --split 1 --train_what ft --epochs 200 --seq_len 5 --num_seq 6 --ds 1 --batch_size 96 --num_workers 32 --gpu 4 5 > $log_file_training
echo "Finished training on HMDB51 split 1 with exit code $?" >> $log_file
date >> $log_file

echo "" >> $log_file

echo "Start training of random network on HMDB51 split 2:" >> $log_file
date >> $log_file
python3 test.py --prefix ut_training_hmdb51_sp_2_ep_200 --dataset hmdb51 --split 2 --train_what ft --epochs 200 --seq_len 5 --num_seq 6 --ds 1 --batch_size 96 --num_workers 32 --gpu 4 5  >> $log_file_training
echo "Finished training on HMDB51 split 2 with exit code $?" >> $log_file
date >> $log_file

echo "" >> $log_file

echo "Start training of random network on HMDB51 split 3:" >> $log_file
date >> $log_file
python3 test.py --prefix ut_training_hmdb51_sp_3_ep_200 --dataset hmdb51 --split 3 --train_what ft --epochs 200 --seq_len 5 --num_seq 6 --ds 1 --batch_size 96 --num_workers 32 --gpu 4 5  >> $log_file_training
echo "Finished training on HMDB51 split 3 with exit code $?" >> $log_file
date >> $log_file
