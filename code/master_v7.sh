PROJ=/data/haiyuez/visteon_cabin_vlm
source /home/haiyuez/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
cd $PROJ/code
pickgpu(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits|awk -F", " '{if($2<3000)print $1}'|head -1; }

# ===== STAGE 1: SUN397 outdoor-landscape domain filter =====
conda activate $PROJ/envs/cabin-vlm
G=$(pickgpu); echo "[stage1] SUN397 filter on GPU$G"
CUDA_VISIBLE_DEVICES=$G python relevance_filter.py \
  --inp $PROJ/data/landscape_sharegpt.json --out $PROJ/data/landscape_outdoor_sharegpt.json \
  --q "Is this an OUTDOOR natural landscape (sky, vegetation, terrain, water, mountains) one could see while driving? Answer strictly yes or no." --tag SUN397_FILTER
echo "STAGE1_DONE"

# ===== STAGE 2: build domain-pure + deduped train_v7 =====
python3 build_train_v7.py
echo "STAGE2_DONE"

# ===== STAGE 3: train v7 =====
cat > train_bootstrap_v7.yaml <<YAML
model_name_or_path: $PROJ/models/Qwen2.5-VL-7B-Instruct
trust_remote_code: true
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
dataset: train_v7
dataset_dir: $PROJ/data
template: qwen2_vl
cutoff_len: 4096
image_max_pixels: 262144
overwrite_cache: true
preprocessing_num_workers: 8
output_dir: $PROJ/models/bootstrap_v7_lora
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
logging_steps: 10
save_steps: 1000
plot_loss: true
report_to: none
YAML
cat > export_bootstrap_v7.yaml <<YAML
model_name_or_path: $PROJ/models/Qwen2.5-VL-7B-Instruct
adapter_name_or_path: $PROJ/models/bootstrap_v7_lora
template: qwen2_vl
trust_remote_code: true
export_dir: $PROJ/models/bootstrap_v7_merged
export_size: 5
export_legacy_format: false
YAML
conda activate $PROJ/envs/llamafactory
G2=$(pickgpu); echo "[stage3] train v7 on GPU$G2"
CUDA_VISIBLE_DEVICES=$G2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True llamafactory-cli train train_bootstrap_v7.yaml
echo "STAGE3_TRAIN_DONE"
G3=$(pickgpu); CUDA_VISIBLE_DEVICES=$G3 llamafactory-cli export export_bootstrap_v7.yaml
echo "V7_ALL_DONE"
