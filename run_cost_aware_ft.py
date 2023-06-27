import os

betas = [0, 0.001, 0.01, 0.1, 1, 10, 100]
weight_types = ['forecast', 'future', 'difference']

for beta in betas:
    for weight_type in weight_types:
        cmd = f'python cost_aware_ft.py --load_path pretrained_unweighted \
            --model_path beta_{beta}_weighted_{weight_type}\
            --n_epochs 150 --cost_weight {beta} --weight_using {weight_type}'
        print(cmd)
        os.system(cmd)
