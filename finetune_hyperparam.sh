# python finetune_double.py --load_path pretrained_unweighted --model_path all_finetuned_unweighted_hist10_mixed_transitions_1e-04 --n_epochs 50 --weight 1 --input_n 10 --lr_ft 1e-04
# python finetune_double.py --load_path pretrained_unweighted --model_path all_finetuned_unweighted_hist10_no_transitions_1e-04 --n_epochs 50 --weight 1 --input_n 10 --transitions 0 --lr_ft 1e-04
# python finetune_double.py --load_path pretrained_unweighted --model_path all_finetuned_unweighted_hist10_only_transitions_1e-04 --n_epochs 50 --weight 1 --input_n 10 --nontransitions 0 --lr_ft 1e-04

# python pretrain.py --input_n 10 --output_n 10 --model_path pretrained_unweighted_output10 --weight 1
# python pretrain.py --input_n 10 --output_n 15 --model_path pretrained_unweighted_output15 --weight 1
# python pretrain.py --input_n 10 --output_n 20 --model_path pretrained_unweighted_output20 --weight 1

# python finetune_double.py --load_path pretrained_wrist6 --model_path all_finetuned_wrist6_hist10_mixed_transitions_1e-04 --n_epochs 50 --weight 6 --input_n 10 --lr_ft 1e-04
# python finetune_double.py --load_path pretrained_wrist6 --model_path all_finetuned_wrist6_hist10_no_transitions_1e-04 --n_epochs 50 --weight 6 --input_n 10 --transitions 0 --lr_ft 1e-04
# python finetune_double.py --load_path pretrained_wrist6 --model_path all_finetuned_wrist6_hist10_only_transitions_1e-04 --n_epochs 50 --weight 6 --input_n 10 --nontransitions 0 --lr_ft 1e-04

python finetune_double.py --load_path pretrained_unweighted_output10 --model_path all_finetuned_unweighted_hist10_output10_mixed_transitions_1e-04 --n_epochs 50 --weight 6 --input_n 10 --output_n 10 --lr_ft 1e-04
python finetune_double.py --load_path pretrained_unweighted_output15 --model_path all_finetuned_unweighted_hist10_output15_mixed_transitions_1e-04 --n_epochs 50 --weight 6 --input_n 10 --output_n 15 --lr_ft 1e-04
python finetune_double.py --load_path pretrained_unweighted_output20 --model_path all_finetuned_unweighted_hist10_output20_mixed_transitions_1e-04 --n_epochs 50 --weight 6 --input_n 10 --output_n 20 --lr_ft 1e-04