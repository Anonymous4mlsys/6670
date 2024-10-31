  python custom_train_glue.py --task_name "rte" \
      --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
      --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --lr 5e-5\
      --train_lora --lora_rank 16 --lora_alpha 128 \
      --save_name adaboost_iteration_2 --epochs 10 --write_results \
      --use_sample_weights --use_sample_weights_dir "rte_iteration_2.npy"