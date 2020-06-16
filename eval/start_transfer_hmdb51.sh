mkdir -p log_files/transfer

log_file=log_files/transfer/_run2_pretrained_nturgbd_cont_transfer_hmdb51_all_splits.log
log_file_training=log_files/transfer/_run2_pretrained_nturgbd_cont_transfer_hmdb51_all_splits_training.log
pretrained_net="/home/dschneider/workspace/DPC/dpc/20200613221038_training_skelcont/nturgbd-128_r18_skelcont_bs40_len30_ds1_train-all/model/model_best_epoch63.pth.tar"

dataset="hmdb51"
seq_len=30
num_seq=1
train_what="ft"
batch_size=30
num_workers=32
gpus="0 3"
print_freq=20

echo "Start of transfer training on HMDB51 split 1:" > $log_file
date >> $log_file
# python3 -u test.py --print_freq $print_freq --prefix pt_nturgbd_cont_transfer_hmdb51_sp_1_ep_200_ft --dataset $dataset --split 1 --train_what $train_what --epochs 200 --seq_len $seq_len --num_seq $num_seq --ds 1 --batch_size $batch_size --num_workers $num_workers --gpu $gpus --pretrain $pretrained_net 2>&1 | tee -a $log_file_training
echo "Finished evaluation training on HMDB51 split 1 with exit code $?" >> $log_file
date >> $log_file

echo "" >> $log_file

echo "Start of evaluation training on HMDB51 split 2:" >> $log_file
date >> $log_file
python3 -u test.py --print_freq $print_freq --prefix run2_pt_nturgbd_cont_transfer_hmdb51_sp_2_ep_200_ft --dataset $dataset --split 2 --train_what $train_what --epochs 200 --seq_len $seq_len --num_seq $num_seq --ds 1 --batch_size $batch_size --num_workers $num_workers --gpu $gpus --pretrain $pretrained_net 2>&1 | tee $log_file_training
echo "Finished evaluation training on HMDB51 split 2 with exit code $?" >> $log_file
date >> $log_file

echo "" >> $log_file

echo "Start of evaluation training on HMDB51 split 3:" >> $log_file
date >> $log_file
python3 -u test.py --print_freq $print_freq --prefix run2_pt_nturgbd_cont_transfer_hmdb51_sp_3_ep_200_ft --dataset $dataset --split 3 --train_what $train_what --epochs 200 --seq_len $seq_len --num_seq $num_seq --ds 1 --batch_size $batch_size --num_workers $num_workers --gpu $gpus --pretrain $pretrained_net  2>&1 | tee -a $log_file_training
echo "Finished evaluation training on HMDB51 split 3 with exit code $?" >> $log_file
date >> $log_file
