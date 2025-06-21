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

# To-do: Klasyfikacja zdjęć MEL (czerniak)

1. **Wydzielenie danych:**
   - Wydzielamy dane dla **MEL** (czerniak).
   - Dzielimy dane na zbiór treningowy oraz testowy.

2. **Augmentacja danych:**
   - Przeprowadzamy augmentację danych, generując około 4k zdjęć do zbioru treningowego.

3. **Trening klasyfikatora:**
   - Trenujemy klasyfikator do klasyfikacji binarnej:
     - `0`: brak nowotworu (bona-fide, około 4k zdjęć)
     - `1`: obecność nowotworu

4. **Szkolenie modeli generatywnych:**
   - Uczymy modele:
     - Dyfuzja
     - GAN
     - VAE
   - Generujemy dane syntetyczne.

5. **Ocena skuteczności klasyfikatora:**
   - Porównujemy wyniki klasyfikatora na danych syntetycznych z danymi autentycznymi.
