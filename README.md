# project-pgm

1. Export your Weights&Biases API Key if you want to track your training and evaluation metrics

```
export WANDB_API_KEY=<api_key>

```

2. Dowload data. You have to be in main directory and run:
```
make all
```
This will run script which will download and prepare data


3. To run training type:
```
python train.py --model <model_name> --variant <variant_name>
```
where:
* ` model_name`: ["cnn", "gan", "vae", "diffusion"] specifies which model to train
* `variant_name`: ["base", "gan", "vae", "diffusion"] is **optional** parameter which is used only for CNN training.

4. To run genrating samples:
```
python generate.py --model <model_name>
```

where:
* ` model_name`: ["gan", "vae", "diffusion"] specifies which model used for generating samples
