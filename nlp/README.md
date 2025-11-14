安装依赖项：

```python
conda create -n theseus python=3.8

conda activate theseus

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
 
cd BERT-of-Theseus

pip install -r requirements.txt

```

用法：

```sh
cd BERT-of-Theseus
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
                                 

```