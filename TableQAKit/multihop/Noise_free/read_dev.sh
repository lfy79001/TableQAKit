# run dev
CUDA_VISIBLE_DEVICES=1 \
python read.py \
    --batch_size 4 \
    --load_dir generation_best \
    --output_dir generation_best \
    --is_train 0 \
    --is_test 0