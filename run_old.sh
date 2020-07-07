### Cautiously use CUDA_LAUNCH_BLOCKING=1. It makes call to single GPU synchronously. Also, it may block the process if used on multiple GPUs due to NCCL
: ${sample_duration:="32"}
: ${sample_size:="224"}
: ${batch_size:="4"}
: ${ckpt_num:="72"}
: ${window_stride:="8"}
: ${model:="resnet"}
: ${simnet_path:=''}
: ${predict_type:='val'}

if [ -z ${PRUN_CPU_RANK+x} ]; then
    echo "Executing for window stride: $window_stride"
else
    # window_stride=`echo "2^$PRUN_CPU_RANK" | bc`
    echo "Executing for window stride: $window_stride, PRUN rank $PRUN_CPU_RANK"
    module load cuda90
    module load nccl/cuda80
fi

#  python3 main.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt --dataset pkummd --n_classes 400 --n_finetune_classes 51 --pretrain_path models/resnet-18-kinetics.pth --ft_begin_index 4 --model resnet --model_depth 18 --resnet_shortcut A --batch_size 16 --n_threads 4 --checkpoint 5 --sample_duration 64

# Train ResNext-101
# python3 main.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt/resnext-101/sample_duration_64/image_size_112/all_windows/non_overlap --dataset pkummd --n_classes 400 --n_finetune_classes 51 --pretrain_path models/resnext-101-64f-kinetics.pth --ft_begin_index 4 --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration 64 --batch_size 8 --n_threads 8 --checkpoint 1 --no_val --resume_path /home/vne500/datasets/PKUMMD/model_ckpt/resnext-101/sample_duration_64/image_size_112/all_windows/non_overlap/save_2.pth

# Train Resnext-101 with larger image size: 224
# python3 main.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt/resnext-101/sample_duration_${sample_duration}/image_size_${sample_size}/ --dataset pkummd --n_classes 400 --n_finetune_classes 51 --pretrain_path models/resnext-101-64f-kinetics.pth --ft_begin_index 4 --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration ${sample_duration} --batch_size ${batch_size} --n_threads 8 --checkpoint 3 --no_val --sample_size ${sample_size} --resume_path /home/vne500/datasets/PKUMMD/model_ckpt/resnext-101/sample_duration_${sample_duration}/image_size_${sample_size}/save_${ckpt_num}.pth

# Predict with sample duration and window_size beign the same. Otherwise, need to form batches and then average the prediction
if [ "${model}" == "resnet" ]; then
echo "Executing model resnet-18..."
python3 predict_window.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt --dataset pkummd --n_classes 51 --ft_begin_index 4 --model resnet --model_depth 18 --resnet_shortcut A --batch_size 1 --n_threads 8 --checkpoint 5 --sample_duration ${sample_duration} --resume_path /home/vne500/datasets/PKUMMD/model_ckpt/resnet-18/sample_duration_${sample_duration}/non_kd_train/image_size_${sample_size}/save_${ckpt_num}.pth --no_train --no_val --test_subset ${predict_type} --test --window_size=${sample_duration} --window_stride=${window_stride} --scores_dump_path scores_dump/resnet-18/sample_duration_${sample_duration}/image_size_${sample_size}/window_${window_stride}/${predict_type}/raw_features --sample_size ${sample_size} --no_cuda_predict --no_softmax_in_test # --resume_path_sim ${simnet_path}

# Predict Resnet-18 with sample duration:16 and window_size larger than 16.
# python3 predict_window.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt --dataset pkummd --n_classes 51 --ft_begin_index 4 --model resnet --model_depth 18 --resnet_shortcut A --batch_size 1 --n_threads 4 --checkpoint 5 --sample_duration 16 --resume_path /home/vne500/datasets/PKUMMD/model_ckpt/resnet-18/sample_duration_16/non_kd_train/save_39.pth --no_train --no_val --test_subset val --test --window_size=64 --window_stride=32 --scores_dump_path scores_dump/resnet-18/sample_duration_16/non_kd_train/window_32/val

# Predict ResNext-101
# : ${sample_duration:="32"}
# :  ${window_stride:="16"}
elif [ "${model}" == "resnext-101" ]; then
echo "Executing model resnet-101..."
python3 predict_window.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt --dataset pkummd --n_classes 51 --ft_begin_index 4 --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration ${sample_duration} --batch_size 8 --n_threads 8 --checkpoint 3 --no_val --resume_path /home/vne500/datasets/PKUMMD/model_ckpt/resnext-101/sample_duration_${sample_duration}/image_size_${sample_size}/save_${ckpt_num}.pth --no_train --test_subset ${predict_type} --test --window_size=${sample_duration} --window_stride=${window_stride} --scores_dump_path scores_dump/resnext-101/sample_duration_${sample_duration}/image_size_${sample_size}/window_${window_stride}/${predict_type} --sample_size ${sample_size} --no_cuda_predict
# Predict ResNext-101 with larger image size: 224 
# python3 predict_window.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt --dataset pkummd --n_classes 51 --ft_begin_index 4 --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration 64 --batch_size 1 --n_threads 8 --no_val --resume_path /home/vne500/datasets/PKUMMD/model_ckpt/resnext-101/image_size_224/save_39.pth --no_train --test_subset val --test --window_size=64 --window_stride=32 --scores_dump_path scores_dump/resnext-101/image_size_224/epoch_39/window_32/val --sample_size 224

# KD_train: Resnet-18
# python3 main.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt/resnet-18/kd_train --dataset pkummd --n_classes 400 --n_finetune_classes 51 --pretrain_path models/resnet-18-kinetics.pth --ft_begin_index 4 --model resnet --model_depth 18 --resnet_shortcut A --batch_size 32 --n_threads 4 --checkpoint 1 --sample_duration 64 --kd_train --teacher_model resnext --teacher_model_depth 101 --teacher_resnet_shortcut B --teacher_resnext_cardinality 32 --teacher_model_path model_ckpt/resnext-101/image_size_112/save_39.pth --teacher_pretrain_path models/resnext-101-64f-kinetics.pth --teacher_batch_size 8 --no_val --resume_path /var/scratch2/vne500/datasets/PKUMMD/model_ckpt/resnet-18/kd_train/save_39.pth

# non KD_train: Resnet-18
# python3 main.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt/resnet-18/sample_duration_${sample_duration}/non_kd_train/image_size_${sample_size}/ --dataset pkummd --n_classes 400 --n_finetune_classes 51 --pretrain_path models/resnet-18-kinetics.pth --ft_begin_index 4 --model resnet --model_depth 18 --resnet_shortcut A --batch_size ${batch_size} --n_threads 8 --checkpoint 3 --sample_duration ${sample_duration} --no_val --sample_size ${sample_size} --resume_path /var/scratch2/vne500/datasets/PKUMMD/model_ckpt/resnet-18/sample_duration_${sample_duration}/non_kd_train/image_size_${sample_size}/save_${ckpt_num}.pth

# non KD_train: Resnet-18 with smaller sample duration: 16
# python3 main.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt/resnet-18/sample_duration_16/non_kd_train --dataset pkummd --n_classes 400 --n_finetune_classes 51 --pretrain_path models/resnet-18-kinetics.pth --ft_begin_index 4 --model resnet --model_depth 18 --resnet_shortcut A --batch_size 32 --n_threads 4 --checkpoint 3 --sample_duration 16 --no_val # --resume_path /var/scratch2/vne500/datasets/PKUMMD/model_ckpt/resnet-18/sample_duration_16/non_kd_train/save_12.pth

# Model validation resnet-18
# echo "Validating model resnet-18..."
# python3 main.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt/resnet-18/sample_duration_${sample_duration}/non_kd_train/image_size_${sample_size}/ --dataset pkummd --n_classes 400 --n_finetune_classes 51 --pretrain_path models/resnet-18-kinetics.pth --ft_begin_index 4 --model resnet --model_depth 18 --resnet_shortcut A --batch_size ${batch_size} --n_threads 8 --checkpoint 3 --sample_duration ${sample_duration} --no_train --sample_size ${sample_size} --resume_path /var/scratch2/vne500/datasets/PKUMMD/model_ckpt/resnet-18/sample_duration_${sample_duration}/non_kd_train/image_size_${sample_size}/save_${ckpt_num}.pth

# python3 main.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt/resnext-101/sample_duration_${sample_duration}/image_size_${sample_size}/first_window --dataset pkummd --n_classes 400 --n_finetune_classes 51 --pretrain_path models/resnext-101-64f-kinetics.pth --ft_begin_index 4 --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration ${sample_duration} --batch_size ${batch_size} --n_threads 8 --checkpoint 3 --no_train --sample_size ${sample_size} --resume_path /home/vne500/datasets/PKUMMD/model_ckpt/resnext-101/sample_duration_${sample_duration}/image_size_${sample_size}/first_window/save_${ckpt_num}.pth

# Convert torch resnet-18 to onxx model
# python3 convert_torch2onnx.py --n_classes 51 --model resnet --model_depth 18 --resnet_shortcut A --sample_duration 64 --resume_path /home/vne500/datasets/PKUMMD/model_ckpt/resnet-18/kd_train/save_40.pth --no_cuda

# Convert torch resnext-101 to onxx model
# python3 convert_torch2onnx.py --n_classes 51 --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration 64 --resume_path /home/vne500/datasets/PKUMMD/model_ckpt/resnext-101/sample_duration_64/image_size_112/save_40.pth --no_cuda

# Train SIMNET
elif [ "${model}" == "simnet" ]; then
echo "Executing simnet.."
python3 main_simnet.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt/simnet/resnet-18/image_size_${sample_size} --dataset pkummd_sim --n_classes 400 --n_finetune_classes 51 --pretrain_path models/resnet-18-kinetics.pth --ft_begin_index 4 --model resnet --model_depth 18 --resnet_shortcut A --batch_size ${batch_size} --n_threads 8 --checkpoint 3 --sample_duration ${sample_duration} --no_val --sample_size ${sample_size} --resume_path /var/scratch2/vne500/datasets/PKUMMD/model_ckpt/resnet-18/sample_duration_${sample_duration}/non_kd_train/image_size_${sample_size}/save_${ckpt_num}.pth --learning_rate 0.001 --resume_path_sim ${simnet_path}
fi
