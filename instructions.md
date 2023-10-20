pip install -r requirements.txt

export PYTHONPATH=<path to parent directory>

python src/pretrain.py --input_n 10 --model_path pretrained_unweighted --weight 1

python src/finetune.py --load_path pretrained_unweighted --model_path all_finetuned_unweighted_with_transitions --n_epochs 20 --weight 1 --input_n 10

python src/cost_aware_finetune.py --load_path all_finetuned_unweighted_hist10_only_transitions_1e-04 --model_path all_finetuned_unweighted_hist10_output25_no_transitiions_costs_1e-04 --n_epochs 150 --input_n 10 --output_n 25 --lr_ft 1e-04 --cost_weight 100