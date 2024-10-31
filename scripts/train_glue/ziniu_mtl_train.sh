# "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "story_cloze" "hellaswag" "winogrande_debiased"
python custom_train_glue_mtl_seq_bn.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "story_cloze" "hellaswag" "winogrande_debiased" \
      --model_key  "meta-llama/Llama-3.2-1B"\
      --devices 1 --batch_size 4 --inference_batch_size 8 --max_length 256 --runs 2 --lr 1e-4\
      --save_name pairwise_lora --epochs 10 --write_results\
      --train_adapter --reduction_factor 128 --use_qadapter\
      --precision "bf16-true" 