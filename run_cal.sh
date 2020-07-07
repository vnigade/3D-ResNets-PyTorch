### Cautiously use CUDA_LAUNCH_BLOCKING=1. It makes call to single GPU synchronously. Also, it may block the process if used on multiple GPUs due to NCCL
: ${sample_duration:="32"}
: ${sample_size:="112"}
: ${batch_size:="8"}
: ${ckpt_num:="72"}
: ${window_stride:="16"}
: ${model:="resnet"}
: ${simnet_path:=''}
: ${predict_type:='val'} # train or val
: ${run_type:='predict'} # train or predict

if [ "${model}" == "resnet" ]; then
echo "Executing resnet.."
python3 main_calibration.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt/calibration --dataset pkummd --n_classes 400 --n_finetune_classes 51 --pretrain_path models/resnet-18-kinetics.pth --ft_begin_index 4 --model resnet --model_depth 18 --resnet_shortcut A --batch_size ${batch_size} --n_threads 8 --checkpoint 3 --sample_duration ${sample_duration} --no_train --sample_size ${sample_size} --resume_path $HOME/datasets/PKUMMD/model_ckpt/resnet-18/sample_duration_${sample_duration}/non_kd_train/image_size_${sample_size}/save_${ckpt_num}.pth --no_softmax_in_test
elif [ "${model}" == "resnext" ]; then
echo "Executing resnext.."
python3 main_calibration.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt/calibration --dataset pkummd --n_classes 400 --n_finetune_classes 51 --pretrain_path models/resnext-101-64f-kinetics.pth --ft_begin_index 4 --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration ${sample_duration} --batch_size ${batch_size} --n_threads 8 --checkpoint 3 --no_train --sample_size ${sample_size} --resume_path $HOME/datasets/PKUMMD/model_ckpt/resnext-101/sample_duration_${sample_duration}/image_size_${sample_size}/save_${ckpt_num}.pth --no_softmax_in_test
fi
