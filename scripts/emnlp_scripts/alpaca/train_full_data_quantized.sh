python custom_train_alpaca.py --model_key "google/gemma-2b"\
    --lr 2e-5 --batch_size 4 --max_length 256 --epochs 10\
    --train_lora --lora_rank 128 --lora_alpha 512\
    --strategy auto --devices 0 --runs 1 --precision "bf16-true" --accumulate 1 --save_every_epoch --save_name test_quantize\
    --use_qlora --optimizer "paged_adamw_8bit"
# google/gemma-2b (2.6)
# ../llama/llama-3/Meta-Llama-3-8B-hf (8.03)
# EleutherAI/gpt-neox-20b 
# google/gemma-2-27b (27.2)

# Total memory: 24576MiB
# Gemma-2B, 8-bit quantized, 8-bit optimizer, batch size 1 (len 2): 2723MiB
# Llama-3-8B, 8-bit quantized, 8-bit optimizer, batch size 1 (len 2): 6841MiB
# GPT-neox-20B, 8-bit quantized, 8-bit optimizer, batch size 1 (len 2): 13827MiB
# 
#
# Length 256
# Length 512
# Length 1024