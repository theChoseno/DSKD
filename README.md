# Direction Sensitivity–based Knowledge Distillation: Optimization-Aware Low‑Rank Knowledge Transfer

### **Introduction**

We propose a Direction Sensitivity-based Knowledge Distillation framework, namely DSKD, to enhance knowledge transfer efficiency in model compression. Given the parameter matrices of teacher and student models, we first quantify the optimization sensitivity of each singular direction through gradient and curvature signals, and then dynamically select the most influential directions at different training stages to construct a direction-weighted low-rank distillation loss. Original paper is "Direction Sensitivity–based Knowledge Distillation: Optimization-Aware Low-Rank Knowledge Transfer".

### Methods

An overview of our DSKD. We apply SVD on the parameter update part of the student model and use the direction sensitivity metric to select the k most sensitive singular directions. Using normalized sensitivity weights, we reconstruct low-rank approximations of both teacher and student models along these directions. The alignment in the sensitive subspace is then measured via MSE loss between the reconstructed matrices. Our approach preserves the most influential update directions and enables fine-grained, direction-aware knowledge transfer.

![overview](/resources/overview.jpg)

### Installation

##### NLP

```
conda create -n theseus python=3.8

conda activate theseus

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
 
cd nlp/BERT-of-Theseus

pip install -r requirements.txt
```

##### CV

```
conda create -n expkd python=3.8

conda activate expkd

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

cd cv
pip install -r requirements.txt
python setup.py develop
```

### Usage

```
### NLP Task ###
python run_glue_SVD_losslayer.py --model_name_or_path 
								 --task_name 
                                 --do_train 
                                 --do_eval 
                                 --do_lower_case 
                                 --data_dir  
                                 --max_seq_length 128 
                                 --per_gpu_train_batch_size 32 
                                 --per_gpu_eval_batch_size 32 
                                 --learning_rate 2e-5 
                                 --save_steps 50 
                                 --num_train_epochs 20 
                                 --output_dir 
                                 --evaluate_during_training 
                                 --replacing_rate 0.3 
                                 --scheduler_type linear 
                                 --scheduler_linear_k 0.0003 
                                 --overwrite_output_dir
                                 --ifsvd
                                 --svd_steps 100
                                 --lambda_size 0.1
                                 --rank_size 32

### CV Task ###
python train_svd.py --cfg configs/cifar100/svd/expkd/res56_res20.yaml
```

