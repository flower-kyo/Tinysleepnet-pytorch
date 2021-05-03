## Tinysleepnet implemented with Pytorch

click to read more about [Tinysleepnet](https://github.com/akaraspt/tinysleepnet).

## Environment

* pytorch >=1.6.0
* tensorboardX
* tensorboard
* scikit-learn

## Create a virtual environment with conda

```python
conda create -n tinysleepnet python=3.6
conda activate tinysleepnet
pip install -r requirements.txt
```

## How to run

1. `python download_sleepedf.py`
2. `python prepare_sleepedf.py`
3. `python trainer.py --db sleepedf --gpu 0 --from_fold 0 --to_fold 19`
4. `python predict.py --config_file config/sleepedf.py --model_dir out_sleepedf/train --output_dir out_sleepedf/predict --log_file out_sleepedf/predict.log --use-best --gpu 0`









