# Dataset Distillation with Pre-trained Models: A Contrastive Approach

This is official code of paper "Dataset Distillation with Pre-trained Models: A Contrastive Approach". 

> **Abstract:** Dataset distillation is a prominent technique that compresses knowledge from a large-scale original dataset into a small synthetic dataset for efficient training. Recent advancements in dataset distillation have demonstrated that pre-trained models can guide the distillation process by providing auxiliary information. However, most methods are limited to support in-distribution pre-trained models, where the label space of pre-trained models must match that of the dataset intended for distillation. We argue that this limitation underestimates the broader potential of out-of-distribution pre-trained models, such as foundational models. To support a flexible and wide range of pre-trained models, we introduce a plug-and-play loss term, namely Contrastive Loss of pre-trained Model (CLoM). Specifically, CLoM contrasts the original example and the synthetic example by some distance measures and treats them as positive pairs if their labels are the same. Equipped with CLoM, we conduct parallel experiments on both in-distribution and out-of-distribution pre-trained models. Extensive experimental evaluations demonstrate the effectiveness of CLoM in enhancing the performance and cross-architecture generalization of synthetic datasets. Furthermore, guided by CLoM, we elucidate the beneficial impact of foundational models on the distillation process, which unlocks the potential of foundational models in dataset distillation.

The **C**ontrastive **L**oss **o**f pre-trained **M**odel (**CLoM**) is defined as:
```math
\mathcal{L}_{\mathrm{CLoM}}(\mathcal{T}, \mathcal{S})=\sum_{\forall \mathcal{B}} \frac{\sum_{i, j} D_{\mathcal{B}^\tau, \mathcal{B}^{\mathcal{S}}} \odot M_{\mathcal{B}^\tau, \mathcal{B}^{\mathcal{S}}}}{\sum_{i, j} D_{\mathcal{B}^\tau, \mathcal{B}^{\mathcal{s}}}},
```
where $`M_{\mathcal{B}^\tau,\mathcal{B}^S}:=O\left(y_{\mathcal{B}^\tau}\right)O\left(y_{\mathcal{B}^S}\right)^T`$ , $`O\left(y_{\mathcal{B}^\tau}\right)`$ is the one-hot encoding of label $`y_{\mathcal{B}^\tau}`$ and $`\odot`$ denotes the Hadamard (element-wise) product of two matrices. 

We employ the Cross-Entropy(CE) distance for in-distribution models and cosine distance for out-of-distribution models respectively:

CE distance:
```math
D_{\mathcal{B}^\tau, \mathcal{B}^{\mathcal{s}}}[i, j]:=\texttt{Dist}_{\mathrm{CE}}\left(\mathcal{B}_i^{\mathcal{T}}, \mathcal{B}_j^{\mathcal{S}}\right):=-O\left(y_{\mathcal{B}_i^\tau}\right)^{\top} \log p^*\left(x_{\mathcal{B}_j^{\mathcal{S}}}\right),
```
where $` p^*\left(x_{\mathcal{B}_j^{\mathcal{S}}}\right)`$ is the probability vector predicted by the pre-trained model $` \theta^*`$ given sample $`x_{\mathcal{B}_j^{\mathcal{S}}}`$.
  
* cosine distance:
```math
D_{\mathcal B^{\mathcal T}, \mathcal B^{\mathcal S}}[i,j] := \texttt{Dist}_{\text{cos}}(\mathcal B^{\mathcal T}_{i}, \mathcal B^{\mathcal S}_{j}):= 1- \frac{F({x_{\mathcal B^{\mathcal T}_{i}})}^\top F({x_{\mathcal B^{\mathcal S}_{j}}})}{\left\|F({x_{\mathcal B^{\mathcal T}_{i}}})\right\|_2\left\|F({x_{\mathcal B^{\mathcal S}_{j}}})\right\|_2}.
```
where  $` F\left(x_{\mathcal{B}_i^\tau}\right) `$ denotes the output vector of either an embedding model or a classification model.

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

## CLoM

Execute following commands to generated synthesis dataset with CLoM on in-distribution pre-trained models with CE distance.

```sh
python methods/DC_DSA_DM/main.py --method DC --dataset CIFAR10 --model ConvNet --ipc 10 --CLoM --CLoM_distance ce --models_pool ConvNet --alphas 1000  --model_num 1 --epoch 150 --source_dataset CIFAR10 --CLoM_batch_size 8196
```

or generated synthesis dataset with CLoM on out-distribution pre-trained models with cosine distance.

```sh
python methods/DC_DSA_DM/main.py --method DC --dataset CIFAR10 --model ConvNet --ipc 10 --CLoM --CLoM_distance cos --models_pool ConvNet --alphas 1000  --model_num 1 --epoch 150 --source_dataset CIFAR100 --CLoM_batch_size 8196
```

* set `--CLoM` to enable CLoM.

* The synthesis dataset will be saved at `./condensed/[dataset]/[method]/...`.

* Use the default path of original dataset `./data/[dataset]` or set `--data_path` to specify the path.

* `--alphas` :the weights of each model architecture.

   `--models_pool` : model architectures

   `--model_num`: number of each architecture(initialization parameters)

   `--epoch`: training epoch the models are at

  `--source_dataset`: the dataset (domain) the models are trained on
  
  `--CLoM_batch_size`: the batch size of CLoM

## Validate synthesis dataset

The following command is an example of validating a specified synthesis dataset.

```bash
python validate.py --normalize_data --dataset CIFAR10 --model ConvNet --dsa --method DC --ipc 10 --synthesis_data_path <specified_path>
```

*  Set ` --save_model` to save model.

