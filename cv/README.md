安装依赖项：

```python
conda create -n expkd python=3.8

conda activate expkd

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

pip install -r requirements.txt
python setup.py develop
```

用法：

```sh
python train_svd.py --cfg configs/cifar100/svd/expkd/res56_res20.yaml
```

