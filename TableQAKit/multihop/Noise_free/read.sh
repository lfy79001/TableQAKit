# run dev
CUDA_VISIBLE_DEVICES=5 \
python read.py \
    --batch_size 4 \
    --load_dir read1 \
    --output_dir read2 \
    --is_train 1 \
    --is_test 0