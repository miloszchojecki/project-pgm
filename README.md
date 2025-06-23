# Synthetic Melanoma Image Generation and Evaluation

This project explores synthetic image generation for melanoma using a variety of deep generative models, including CNN-based, GANs, VAEs, and diffusion models. The primary goal is to assess how well these generated images can support melanoma classification tasks—specifically, distinguishing melanoma from non-melanoma samples.

We evaluate the models both qualitatively (via sample generation) and quantitatively (using the FID score and downstream classification performance).

**Configuration Management:** All training parameters, model settings, and paths are managed through the `params.yaml` file located in the project root directory.

**Results & Models:** Trained models, generated samples, and evaluation results are stored and available for download at [drive](https://drive.google.com/drive/folders/1FIHiP5R8WrJW6QlwJS8YXTMKrga_Xri_?usp=sharing).



---

## 1. Optional: Enable Weights & Biases Logging

To track training and evaluation metrics with [Weights & Biases](https://wandb.ai), export your API key:

```bash
export WANDB_API_KEY=<your_api_key>
```
Alternatively, you can disable logging by setting `log: False` in the `params.yaml` file.

---

## 2. Download and Prepare the Dataset

From the main project directory, run:

```bash
make all
```

This will execute a script that downloads and prepares the dataset for training.

---

## 3. Train a Model

Run the following command to train a model:

```bash
python train.py --model <model_name> --variant <variant_name>
```

**Arguments:**

* `--model`: Required. Choose from `cnn`, `gan`, `vae`, `diffusion`.
* `--variant`: Optional. Only applicable when `--model cnn`. Choose from `base`, `gan`, `vae`, `diffusion`.

---

## 4. Generate Samples

To generate samples with a trained model:

```bash
python generate.py --model <model_name>
```

**Arguments:**

* `--model`: Required. Choose from `gan`, `vae`, `diffusion`.

---

## 5. Evaluate with FID

To compute the Fréchet Inception Distance (FID) score:

```bash
python fid.py --model <model_name>
```

**Arguments:**

* `--model`: Required. Choose from `gan`, `vae`, `diffusion`.