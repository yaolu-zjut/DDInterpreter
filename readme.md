# Can pre-trained models assist in dataset distillation?

This is official code of paper "Can pre-trained models assist in dataset distillation?". 

> **Abstract:** Dataset Distillation (DD) is a prominent technique that encapsulates knowledge from a large-scale original dataset into a small synthetic dataset for efficient training. Meanwhile, Pre-trained Models (PTMs) function as knowledge repositories, containing extensive information from the original dataset. This naturally raises a question: Can PTMs effectively transfer knowledge to synthetic datasets, guiding DD accurately? To this end, we conduct preliminary experiments, confirming the contribution of PTMs to DD. Afterwards, we systematically study different options in PTMs, including initialization parameters, model architecture, training epoch and domain knowledge, revealing that: 1) Increasing model diversity enhances the performance of synthetic datasets; 2) Sub-optimal models can also assist in DD and outperform well-trained ones in certain cases; 3) Domain-specific PTMs are not mandatory for DD, but a reasonable domain match is crucial. Finally, by selecting optimal options, we significantly improve the cross-architecture generalization over baseline DD methods. We hope our work will facilitate researchers to develop better DD techniques. 

## Setup

Install packages in the `requirements.txt`.

## Train models on original dataset

The following command will train 10 ConvNet models on CIFAR-10.

```bash
python train_original_models.py --normalize_data --dataset CIFAR10 --model ConvNet --num 10
```

* Set `--num` to point how many models needed to be trained , and the pretrained models will be saved at `./pretrained_model/[dataset]/original/[model]/...`.
*  Use the default path of original dataset `./data/[dataset]` or set `--data_path` to specify the path.
*  Change `--model` to train models with different architectures, such as AlexNet, VGG11, ResNet18, et.

## CLoM & CCLoM

Execute following commands to generated synthesis dataset with **CLoM**(**C**lassification **L**oss **o**f pre-trained **M**odel).

```sh
python method/DC_DSA_DM/main.py --method DC --dataset CIFAR10 --model ConvNet --ipc 10 --CLoM --models_pool ConvNet --alphas 1000  --model_num 1 --epoch 150
```

or generated synthesis dataset with **CCLoM**(**C**ontrastive **C**lassification **L**oss of pre-trained **M**odel) by:

```sh
python method/DC_DSA_DM/main.py --method DC --dataset CIFAR10 --model ConvNet --ipc 10 --CCLoM --models_pool ConvNet --alphas 1000  --model_num 1 --epoch 150 --source_dataset CIFAR100 --CCLoM_batch_size 8196
```

* set `--CLoM` or `--CCLoM` to enable corresponding loss item.

* The synthesis dataset will be saved at `./condensed/[dataset]/[method]/IPC[ipc]/...`.

* Use the default path of original dataset `./data/[dataset]` or set `--data_path` to specify the path.

* `--alphas` :the weights of CLoM/CCLoM.

   `--models_pool` : model architectures

   `--model_num`: number of each architecture(initialization parameters)

   `--epoch`: training epoch the models are at

  `--source_dataset`: the dataset(domain) the models are trained on

## Validate synthesis dataset

The following command is an example of validating a specified synthesis dataset.

```bash
python validate.py --normalize_data --dataset CIFAR10 --model ConvNet --dsa --method DC --ipc 10 --synthesis_data_path <specified_path>
```

*  Set ` --save_model` to save model.

# Thanks for the support
[![Stargazers repo roster for @yaolu-zjut/DDInterpreter](https://reporoster.com/stars/yaolu-zjut/DDInterpreter)](https://github.com/yaolu-zjut/DDInterpreter/stargazers)

# Cite Our Paper
If you use this code in your research, please cite our paper.
```bash
@misc{lu2023pretrained,
      title={Can pre-trained models assist in dataset distillation?}, 
      author={Yao Lu and Xuguang Chen and Yuchen Zhang and Jianyang Gu and Tianle Zhang and Yifan Zhang and Xiaoniu Yang and Qi Xuan and Kai Wang and Yang You},
      year={2023},
      eprint={2310.03295},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
