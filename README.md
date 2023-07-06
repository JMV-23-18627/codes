# Codes of manuscript JMV-23-18627

## Training

1. Path setting

```python
config.model_save_path = '/home/xxxx/works/COVID-19_LTAP/checkpoints/'
data_root = '/home/xxxx/works/COVID-19_LTAP/data/'
static_file = data_root + 'your_static_data.pkl'
CT_file = data_root + 'your_CT_data.pkl'
```

2. Copy the your dataset to the data_root.

3. Run

```sh
CUDA_VISIBLE_DEVICES=0 python train_val.py
```

It will save the models in ./checkpionts/.

## Testing

1. Path setting

```python
data_root = '/home/xxxx/works/COVID-19_LTAP/data/'
static_file = data_root + 'samples_clinical.pkl'
CT_file = data_root + 'samples_CT.pkl'
config.pretrained_model_path = '/home/xxxx/works/COVID-19/checkpointpre-trained_LTAP_model.pkl'
```

2. Copy the pre-trained model to ./checkpionts/.

3. Evaluate

```sh
CUDA_VISIBLE_DEVICES=0 python test.py
```
