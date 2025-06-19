# project-pgm

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
