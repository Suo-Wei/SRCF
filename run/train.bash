save_name=SRCF
gpu=0
CUDA_VISIBLE_DEVICES=$gpu nohup python -u src/train.py \
--model_name $save_name \
--part 6 \
--lr 0.0005 \
--dataset CUHK-PEDES \
--epoch 60 \
--dataroot dataset/CUHK-PEDES/ \
--class_num 11000 \
--vocab_size 5000 \
--feature_length 1024 \
--mode train \
--batch_size 32 \
--cr_beta 0.1

