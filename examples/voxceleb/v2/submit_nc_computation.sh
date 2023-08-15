#!/bin/bash
#SBATCH -J wespeaker_xvec_extraction
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 120:00
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:0

exp_dir=exp/XVEC-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150
gpus="[0,1]"

for epoch in 125 100; do
    echo "Computing NC1 for epoch ${epoch} ..."
    python compute_nc/compute_nc_from_embeddings.py \
        --embed_scp ${exp_dir}/embeddings_${epoch}/vox2_dev/xvector.scp \
        --checkpoint_path ${exp_dir}/models/model_${epoch}.pt
    sleep 10s
done
