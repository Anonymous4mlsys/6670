python custom_train_shakespeare.py --num_tasks 200\
    --model_key "meta-llama/Llama-3.2-1B"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --lr 2e-5 --precision "bf16-true"\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --use_qlora --optimizer "paged_adamw_32bit"\
    --save_name shakespeare_task_200 --epochs 10 --write_results

for i in {100..200}
do
python custom_train_shakespeare.py --num_tasks 200 --task_idxes $i\
    --model_key "meta-llama/Llama-3.2-1B"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --lr 2e-5 --precision "bf16-true"\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --use_qlora --optimizer "paged_adamw_32bit"\
    --save_name shakespeare --epochs 10 --write_results
done