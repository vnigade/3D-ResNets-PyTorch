### Cautiously use CUDA_LAUNCH_BLOCKING=1. It makes call to single GPU synchronously. Also, it may block the process if used on multiple GPUs due to NCCL
: ${sample_duration:="16"}
: ${sample_size:="224"}
: ${batch_size:="16"}
: ${ckpt_num:="288"}
: ${window_stride:="4"}
: ${model:="$1"}
: ${simnet_path:=''}
: ${predict_type:='val'} # train or val
: ${run_type:='predict'} # train or predict

#  python3 main.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt --dataset pkummd --n_classes 400 --n_finetune_classes 51 --pretrain_path models/resnet-18-kinetics.pth --ft_begin_index 4 --model resnet --model_depth 18 --resnet_shortcut A --batch_size 16 --n_threads 4 --checkpoint 5 --sample_duration 64

# Train ResNext-101
# python3 main.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt/resnext-101/sample_duration_64/image_size_112/all_windows/non_overlap --dataset pkummd --n_classes 400 --n_finetune_classes 51 --pretrain_path models/resnext-101-64f-kinetics.pth --ft_begin_index 4 --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration 64 --batch_size 8 --n_threads 8 --checkpoint 1 --no_val --resume_path ${HOME}/datasets/PKUMMD/model_ckpt/resnext-101/sample_duration_64/image_size_112/all_windows/non_overlap/save_2.pth

# Train Resnext-101 with larger image size: 224
if [[ "${model}" == "resnext-101" && "${run_type}" == "train" ]]; then
echo "Training model resnext-101..."
python3 main.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd_view.json --result_path model_ckpt/resnext-101/sample_duration_${sample_duration}/image_size_${sample_size}/cross_view --dataset pkummd --n_classes 400 --n_finetune_classes 51 --pretrain_path models/resnext-101-64f-kinetics.pth --ft_begin_index 4 --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration ${sample_duration} --batch_size ${batch_size} --n_threads 8 --checkpoint 3 --no_val --sample_size ${sample_size} --n_epochs 300 # --resume_path $HOME/datasets/PKUMMD/model_ckpt/resnext-101/sample_duration_${sample_duration}/image_size_${sample_size}/cross_view/save_${ckpt_num}.pth

# Predict with sample duration and window_size beign the same. Otherwise, need to form batches and then average the prediction
elif [[ "${model}" == "resnet" && "${run_type}" == "predict" ]]; then
echo "Predicting model resnet-18..."
python3 predict_window.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd_view.json --result_path model_ckpt --dataset pkummd --n_classes 51 --ft_begin_index 4 --model resnet --model_depth 18 --resnet_shortcut A --batch_size 1 --n_threads 8 --checkpoint 5 --sample_duration ${sample_duration} --resume_path $HOME/datasets/PKUMMD/model_ckpt/resnet-18/sample_duration_${sample_duration}/non_kd_train/image_size_${sample_size}/cross_view/save_${ckpt_num}.pth --no_train --no_val --test_subset ${predict_type} --test --window_size=${sample_duration} --window_stride=${window_stride} --scores_dump_path scores_dump/resnet-18/sample_duration_${sample_duration}/image_size_${sample_size}/window_${window_stride}/cross_view/${predict_type}/raw_features/ --sample_size ${sample_size} --no_softmax_in_test # --resume_path_sim ${simnet_path}

# Predict Resnet-18 with sample duration:16 and window_size larger than 16.
# python3 predict_window.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt --dataset pkummd --n_classes 51 --ft_begin_index 4 --model resnet --model_depth 18 --resnet_shortcut A --batch_size 1 --n_threads 4 --checkpoint 5 --sample_duration 16 --resume_path $HOME/datasets/PKUMMD/model_ckpt/resnet-18/sample_duration_16/non_kd_train/save_39.pth --no_train --no_val --test_subset val --test --window_size=64 --window_stride=32 --scores_dump_path scores_dump/resnet-18/sample_duration_16/non_kd_train/window_32/val

# Predict ResNext-101
# : ${sample_duration:="32"}
# :  ${window_stride:="16"}
elif [[ "${model}" == "resnext-101" && "${run_type}" == "predict" ]]; then
echo "Predicting model resnext-101..."
python3 predict_window.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd_view.json --result_path model_ckpt --dataset pkummd --n_classes 51 --ft_begin_index 4 --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration ${sample_duration} --batch_size 8 --n_threads 8 --checkpoint 3 --no_val --resume_path $HOME/datasets/PKUMMD/model_ckpt/resnext-101/sample_duration_${sample_duration}/image_size_${sample_size}/cross_view/save_${ckpt_num}.pth --no_train --test_subset ${predict_type} --test --window_size=${sample_duration} --window_stride=${window_stride} --scores_dump_path scores_dump/resnext-101/sample_duration_${sample_duration}/image_size_${sample_size}/window_${window_stride}/cross_view/${predict_type}/ --sample_size ${sample_size}
# Predict ResNext-101 with larger image size: 224 
# python3 predict_window.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt --dataset pkummd --n_classes 51 --ft_begin_index 4 --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration 64 --batch_size 1 --n_threads 8 --no_val --resume_path $HOME/datasets/PKUMMD/model_ckpt/resnext-101/image_size_224/save_39.pth --no_train --test_subset val --test --window_size=64 --window_stride=32 --scores_dump_path scores_dump/resnext-101/image_size_224/epoch_39/window_32/val --sample_size 224

# non KD_train: Resnet-18
elif [[ "${model}" == "resnet" && "${run_type}" == "train" ]]; then
echo "Training model resnet-18..."
python3 main.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd_view.json --result_path model_ckpt/resnet-18/sample_duration_${sample_duration}/non_kd_train/image_size_${sample_size}/cross_view/ --dataset pkummd --n_classes 400 --n_finetune_classes 51 --pretrain_path models/resnet-18-kinetics.pth --ft_begin_index 4 --model resnet --model_depth 18 --resnet_shortcut A --batch_size ${batch_size} --n_threads 8 --checkpoint 3 --sample_duration ${sample_duration} --no_val --n_epochs 300 --sample_size ${sample_size} --resume_path $HOME/datasets/PKUMMD/model_ckpt/resnet-18/sample_duration_${sample_duration}/non_kd_train/image_size_${sample_size}/cross_view/save_${ckpt_num}.pth

elif [[ "${model}" == "resnet-kd" && "${run_type}" == "train" ]]; then
echo "KD Training model resnet-18..."
python3 main.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt/resnet-18/sample_duration_${sample_duration}/kd_train/image_size_${sample_size}/ --dataset pkummd --n_classes 400 --n_finetune_classes 51 --pretrain_path models/resnet-18-kinetics.pth --ft_begin_index 4 --model resnet --model_depth 18 --resnet_shortcut A --batch_size ${batch_size} --n_threads 8 --checkpoint 3 --sample_duration ${sample_duration} --kd_train --teacher_model resnext --teacher_model_depth 101 --teacher_resnet_shortcut B --teacher_resnext_cardinality 32 --teacher_model_path model_ckpt/resnext-101/sample_duration_${sample_duration}/image_size_${sample_size}/save_288.pth --teacher_pretrain_path models/resnext-101-64f-kinetics.pth --teacher_batch_size ${batch_size} --teacher_sample_size ${sample_size} --no_train --sample_size ${sample_size} --n_epochs 300 --resume_path $HOME/datasets/PKUMMD/model_ckpt/resnet-18/sample_duration_${sample_duration}/kd_train/image_size_${sample_size}/save_${ckpt_num}.pth

# Model validation resnet-18
# echo "Validating model resnet-18..."
# python3 main.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt/resnet-18/sample_duration_${sample_duration}/non_kd_train/image_size_${sample_size}/ --dataset pkummd --n_classes 400 --n_finetune_classes 51 --pretrain_path models/resnet-18-kinetics.pth --ft_begin_index 4 --model resnet --model_depth 18 --resnet_shortcut A --batch_size ${batch_size} --n_threads 8 --checkpoint 3 --sample_duration ${sample_duration} --no_train --sample_size ${sample_size} --resume_path $HOME/datasets/PKUMMD/model_ckpt/resnet-18/sample_duration_${sample_duration}/non_kd_train/image_size_${sample_size}/save_${ckpt_num}.pth

# python3 main.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt/resnext-101/sample_duration_${sample_duration}/image_size_${sample_size}/first_window --dataset pkummd --n_classes 400 --n_finetune_classes 51 --pretrain_path models/resnext-101-64f-kinetics.pth --ft_begin_index 4 --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration ${sample_duration} --batch_size ${batch_size} --n_threads 8 --checkpoint 3 --no_train --sample_size ${sample_size} --resume_path $HOME/datasets/PKUMMD/model_ckpt/resnext-101/sample_duration_${sample_duration}/image_size_${sample_size}/first_window/save_${ckpt_num}.pth

# Convert torch resnet-18 to onxx model
# python3 convert_torch2onnx.py --n_classes 51 --model resnet --model_depth 18 --resnet_shortcut A --sample_duration 64 --resume_path $HOME/datasets/PKUMMD/model_ckpt/resnet-18/kd_train/save_40.pth --no_cuda

# Convert torch resnext-101 to onxx model
# python3 convert_torch2onnx.py --n_classes 51 --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration 64 --resume_path $HOME/datasets/PKUMMD/model_ckpt/resnext-101/sample_duration_64/image_size_112/save_40.pth --no_cuda

# Train SIMNET
elif [ "${model}" == "simnet" ]; then
echo "Executing simnet.."
python3 main_simnet.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd_view.json --result_path model_ckpt/simnet/resnet-18/sample_duration_${sample_duration}/image_size_${sample_size}/bce_loss/cross_view --dataset pkummd_sim --n_classes 400 --n_finetune_classes 51 --pretrain_path models/resnet-18-kinetics.pth --ft_begin_index 4 --model resnet --model_depth 18 --resnet_shortcut A --batch_size ${batch_size} --n_threads 8 --checkpoint 3 --sample_duration ${sample_duration} --no_val --sample_size ${sample_size} --resume_path $HOME/datasets/PKUMMD/model_ckpt/resnet-18/sample_duration_${sample_duration}/non_kd_train/image_size_${sample_size}/cross_view/save_${ckpt_num}.pth --learning_rate 0.001 --resume_path_sim ~/datasets/PKUMMD/model_ckpt/simnet/resnet-18/sample_duration_${sample_duration}/image_size_${sample_size}/bce_loss/cross_view/save_84.pth

elif [ "${model}" == "simnet-mobilenet" ]; then
echo "Executing simnet-mobilenet.."
python3 main_simnet.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt/simnet/mobilenet/sample_duration_${sample_duration}/image_size_${sample_size}/bce_loss/ --dataset pkummd_sim --n_classes 600 --n_finetune_classes 51 --pretrain_path models/kinetics_mobilenet_1.0x_RGB_${sample_duration}_best.pth --ft_begin_index 4 --model mobilenet --model_depth 1 --batch_size ${batch_size} --n_threads 8 --checkpoint 3 --sample_duration ${sample_duration} --no_val --sample_size ${sample_size} --resume_path $HOME/datasets/PKUMMD/model_ckpt/mobilenet/sample_duration_${sample_duration}/image_size_${sample_size}/save_${ckpt_num}.pth --learning_rate 0.001 # --resume_path_sim ${simnet_path}

# Training MobileNet for early discard
elif [[ "${model}" == "mobilenet-early-discard" && "${run_type}" == "train" ]]; then
echo "Training model mobilenet-early-discard..."
sample_size=112
python3 main_early_discard.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt/${model}/sample_duration_${sample_duration}/image_size_${sample_size}/ --dataset pkummd_ed --n_classes 600 --n_finetune_classes 1 --pretrain_path models/kinetics_mobilenet_1.0x_RGB_${sample_duration}_best.pth --ft_begin_index 4 --model mobilenet --model_depth 1 --batch_size ${batch_size} --n_threads 8 --checkpoint 3 --sample_duration ${sample_duration} --no_val --sample_size ${sample_size} --n_epochs 300 --learning_rate 0.01 # --resume_path $HOME/datasets/PKUMMD/model_ckpt/${model}/sample_duration_${sample_duration}/image_size_${sample_size}/save_${ckpt_num}.pth

elif [[ "${model}" == "mobilenet-early-discard" && "${run_type}" == "predict" ]]; then
echo "Predicting model mobilenet-early-discard..."
sample_size=112
python3 main_early_discard.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt/${model}/sample_duration_${sample_duration}/image_size_${sample_size}/ --dataset pkummd_ed --n_classes 600 --n_finetune_classes 1 --pretrain_path models/kinetics_mobilenet_1.0x_RGB_${sample_duration}_best.pth --ft_begin_index 4 --model mobilenet --model_depth 1 --batch_size ${batch_size} --n_threads 8 --checkpoint 3 --sample_duration ${sample_duration} --no_train --sample_size ${sample_size} --n_epochs 300 --learning_rate 0.01 --resume_path $HOME/datasets/PKUMMD/model_ckpt/${model}/sample_duration_${sample_duration}/image_size_${sample_size}/save_${ckpt_num}.pth

elif [[ "${model}" == "mobilenet" && "${run_type}" == "train" ]]; then
echo "Training model mobilenet..."
sample_size=112
PROG="main.py"
python3 ${PROG} --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt/${model}/sample_duration_${sample_duration}/image_size_${sample_size}/ --dataset pkummd --n_classes 600 --n_finetune_classes 51 --pretrain_path models/kinetics_mobilenet_1.0x_RGB_${sample_duration}_best.pth --ft_begin_index 4 --model mobilenet --model_depth 1  --batch_size ${batch_size} --n_threads 8 --checkpoint 3 --sample_duration ${sample_duration} --no_val --sample_size ${sample_size} --train_crop random --n_epochs 300 --learning_rate 0.01 --resume_path $HOME/datasets/PKUMMD/model_ckpt/${model}/sample_duration_${sample_duration}/image_size_${sample_size}/save_${ckpt_num}.pth

elif [[ "${model}" == "mobilenet" && "${run_type}" == "predict" ]]; then
echo "Predicting model mobilenet..."
sample_size=112
PROG="predict_window.py"
python3 ${PROG} --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt/${model}/sample_duration_${sample_duration}/image_size_${sample_size}/ --dataset pkummd --n_classes 600 --n_finetune_classes 51 --pretrain_path models/kinetics_mobilenet_1.0x_RGB_${sample_duration}_best.pth --ft_begin_index 4 --model mobilenet --model_depth 1  --batch_size ${batch_size} --n_threads 8 --sample_duration ${sample_duration} --no_train --no_val --test_subset ${predict_type} --test --window_size=${sample_duration} --window_stride=${window_stride} --sample_size ${sample_size} --resume_path $HOME/datasets/PKUMMD/model_ckpt/${model}/sample_duration_${sample_duration}/image_size_${sample_size}/save_${ckpt_num}.pth --scores_dump_path scores_dump/${model}/sample_duration_${sample_duration}/image_size_${sample_size}/window_${window_stride}/${predict_type}/raw_features/ --no_softmax_in_test

elif [[ "${model}" == "mobilenetv2" && "${run_type}" == "train" ]]; then
echo "Training model mobilenetv2..."
sample_size=112
python3 main_mobilenet.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt/${model}/sample_duration_${sample_duration}/image_size_${sample_size}/ --dataset pkummd --n_classes 600 --n_finetune_classes 51 --pretrain_path models/kinetics_mobilenetv2_1.0x_RGB_${sample_duration}_best.pth --ft_begin_index 4 --model mobilenetv2 --model_depth 1  --batch_size ${batch_size} --n_threads 8 --checkpoint 3 --sample_duration ${sample_duration} --no_val --sample_size ${sample_size} --train_crop random --n_epochs 300 --learning_rate 0.01 #--resume_path $HOME/datasets/PKUMMD/model_ckpt/resnet-18/sample_duration_${sample_duration}/non_kd_train/image_size_${sample_size}/save_${ckpt_num}.pth
fi
