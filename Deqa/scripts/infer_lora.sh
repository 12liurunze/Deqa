export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH=./:$PYTHONPATH

python /home/lrz/Deqa/src/evaluate/iqa_eval.py \
	--level-names excellent good fair poor bad \
	--model-path /home/lrz/Deqa/output/model/deqa_lora_overall/ \
	--model-base /home/lrz/Deqa/model_weight/ \
	--save-dir /home/lrz/Deqa/output/model/deqa_lora_overall/ \
	--preprocessor-path /home/lrz/Deqa/preprocessor/ \
	--root-dir /home/lrz/Q-Align-main/playground/DIQA-5000_phase1/val/res/ \
	--meta-paths /home/lrz/Q-Align-main/playground/data/converted_dataset_val.json
