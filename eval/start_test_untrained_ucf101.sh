mkdir -p log_files

log_file=log_files/random_trained_test_ucf101_all_splits.log
log_file_training=log_files/random_trained_test_ucf101_all_splits_training.log

network_split_1=""
network_split_2=""
network_split_3=""


echo "Start test of random network trained on UCF101 train split 1:" > $log_file
date >> $log_file
python3 test.py --dataset ucf101 --split 1 --seq_len 5 --num_seq 6 --ds 1 --num_workers 32 --gpu 4 --test $network_split_1 > log_file_training
echo "Finished testing UCF101 split 1 with exit code $?" >> $log_file
date >> $log_file
echo "" >> $log_file

echo "Start test of random network trained on UCF101 train split 2:" >> $log_file
date >> $log_file
python3 test.py --dataset ucf101 --split 2 --seq_len 5 --num_seq 6 --ds 1 --num_workers 32 --gpu 4 --test $network_split_2 >> log_file_training
echo "Finished testing UCF101 split 2 with exit code $?" >> $log_file
date >> $log_file
echo "" >> $log_file

echo "Start test of random network trained on UCF101 train split 3:" >> $log_file
date >> $log_file
python3 test.py --dataset ucf101 --split 3 --seq_len 5 --num_seq 6 --ds 1 --num_workers 32 --gpu 4 --test $network_split_3 >> log_file_training
echo "Finished testing UCF101 split 3 with exit code $?" >> $log_file
date >> $log_file
echo "" >> $log_file

