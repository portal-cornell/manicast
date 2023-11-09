<!-- ## ManiCast: Collaborative Manipulation with Cost-Aware Human Forecasting

<a href="https://portal-cornell.github.io/manicast/">Website</a>

<a href="https://kushal2000.github.io/">Kushal Kedia</a>,
<a href="https://portfolio-pdan101.vercel.app/">Prithwish Dan</a>,
Atiksh Bhardwaj,
<a href="https://www.sanjibanchoudhury.com/">Sanjiban Choudhury</a> -->


## ManiCast: Collaborative Manipulation with Cost-Aware Human Forecasting

**[`Website`](https://portal-cornell.github.io/manicast) | [`Paper`](https://arxiv.org/abs/2310.13258)**

This is a repository containing datasets, visualizations and model checkpoints for the CoRL 2023 paper:\
**ManiCast: Collaborative Manipulation with Cost-Aware Human Forecasting**
<br>
<a href="https://kushal2000.github.io/">Kushal Kedia</a>,
<a href="https://portfolio-pdan101.vercel.app/">Prithwish Dan</a>,
Atiksh Bhardwaj,
<a href="https://www.sanjibanchoudhury.com/">Sanjiban Choudhury</a>

### Real-World Collaborative Manipulation
<table border="0">
 <tr align="center">
    <td><img width="250" height="300" src="docs/stirring_loweres.gif" alt>
    <em>Reactive Stirring</em></td>
    <td><img width="250" height="300" src="docs/handovergif_lowerres.gif" alt>
    <em>Object Handover</em></td>
    <td><img width="250" height="300" src="docs/tableset_lowerres.gif" alt>
    <em>Collaborative Table Setting</em></td>
 </tr>
</table>

Our framework <b>ManiCast</b>
learns <b>cost-aware human motion forecasts</b> and <b>plans with such forecasts</b>
for <b>collaborative manipulation</b> tasks.

<table align="center" border="0">
 <tr align="center">
    <td><img width="350" height="350" src="docs/react_legend.gif" alt><br>
<!--     <em>Reactive Stirring</em></td> -->
    <td><img width="350" height="350" src="docs/handover_legend.gif" alt><br>
<!--     <em>Object Handover</em></td> -->
 </tr>
</table>
<!--     <td><img width="250" height="250" src="docs/table_legend.gif" alt>
    <em>Collaborative Table Setting</em></td> -->
At train time, we fine-tune pre-trained 
human motion forecasting models on task specific datasets by upsampling 
transition points and upweighting joint dimensions that dominate the cost 
of the robot's planned trajectory. At inference time, we feed these forecasts 
into a model predictive control (MPC) planner to compute robot plans that 
are <b>reactive</b> and keep a <b>safe distance</b> from the human.

### Setup

Setup environments following the [SETUP.md](docs/SETUP.md)

We release checkpoints of all the [models](model_checkpoints) used in our paper. The next two sections provide instructions on how to use these models.

### Visualization

Play any data episode via any model.
```
python eval/comad_visualization.py --data_dir {handover, reactive_stirring, table_setting} --visualize_from {train, val, test} --ep_num <EPISODE_NUMBER> --load_path <MODEL_NAME>
```
Python notebook demo through eval/comad_visualization.ipynb.

### Evaluation

Generate evaluation metrics on Object Handovers.
```
python eval/handover.py --ep_num 2
```

Generate evaluation metrics on Reactive Stirring.
```
python eval/reactive_stirring.py --ep_num 4
```

Generate evaluation metrics on CoMaD Dataset.
```
python eval/test_comad.py
```

### Training

Pretrain model on large-scale data (requires following [SETUP.md](docs/SETUP.md) to install AMASS data).
```
python src/pretrain.py --weight 1
```

Finetune the above models using the ManiCast framework.
Add `--load_path default/<epoch num>` to load a model trained with `pretrain.py`. Upweighting wrist/hand joints can be done easily through the `--weight` command line argument.
```
python src/finetune.py --weight 1 `--load_path default/<epoch num>`
```



<!-- ### Work in Progress -->

### Acknowledgement

This repository borrows code from [STS-GCN](https://github.com/FraLuca/STSGCN).

### Citation

```bibtex
@inproceedings{kedia2023manicast,
    title={ManiCast: Collaborative Manipulation with Cost-Aware Human Forecasting},
    author={Kushal Kedia and Prithwish Dan and Atiksh Bhardwaj and Sanjiban Choudhury},
    booktitle={7th Annual Conference on Robot Learning},
    year={2023},
    url={https://openreview.net/forum?id=rxlokRzNWRq}
}   
```
