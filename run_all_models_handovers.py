import os

models = ["pretrained_unweighted", 
          "no_pretraining",
          "all_finetuned_unweighted_hist10_no_transitions_1e-04",
          "all_finetuned_unweighted_hist10_only_transitions_1e-04",
          "all_finetuned_unweighted_hist10_mixed_transitions_1e-04",
          "all_finetuned_wrist6ft_hist10_mixed_transitions_1e-04"]

##chopping_mixing_2
chopping_mixing_2_windows = [(580,770),
           (2700,2800),
           (3100,3300),
           (6400,6600),
           (6750,6950),
           ]

##chopping_stirring_0
chopping_stirring_0_windows = [(580,770),
           (2800,2970),
           (3200,3300),
           (6450,6700),
           (6150,6350),
           ]


# for model in models:
#     print("========================", model)
#     for window in chopping_mixing_2_windows:
#         start, end = window
#         os.system(f"python goal_detection.py \
#                 --load_path {model} \
#                 --activity chopping_mixing --ep_num 2 --threshold 0.10 \
#                 --start_frame {start} --end_frame {end}")
    
#     for window in chopping_stirring_0_windows:
#         start, end = window
#         os.system(f"python goal_detection.py \
#                 --load_path {model} \
#                 --activity chopping_stirring --ep_num 0 --threshold 0.10 \
#                 --start_frame {start} --end_frame {end}")
        
for model in models:
    print("======================== CVM")
    for window in chopping_mixing_2_windows:
        start, end = window
        os.system(f"python goal_detection.py \
                --load_path {model} \
                --activity chopping_mixing --ep_num 2 --threshold 0.10 \
                --start_frame {start} --end_frame {end} --prediction_method cvm")
    
    for window in chopping_stirring_0_windows:
        start, end = window
        os.system(f"python goal_detection.py \
                --load_path {model} \
                --activity chopping_stirring --ep_num 0 --threshold 0.10 \
                --start_frame {start} --end_frame {end} --prediction_method cvm")
    break