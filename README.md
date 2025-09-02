**Predyktor Dominujących Kolorów**
Projekt wykorzystujący sieć konwolucyjną PyTorch do przewidywania 5 dominujących kolorów na podstawie obrazu.  
Z czego składa się projekt?  
* train.py: Skrypt odpowiedzialny za trenowanie sieci neuronowej.  
* gui.py: Aplikacja z interfejsem graficznym (GUI) do testowania wytrenowanych modeli na dowolnych obrazach.  
* images/: Folder, w którym należy umieścić obrazy do treningu.  
* labels.txt: Plik tekstowy zawierający etykiety, czyli nazwy plików i odpowiadające im 5 dominujących kolorów w formacie HEX.  
* models/: Folder, w którym skrypt treningowy automatycznie zapisuje wytrenowane modele.  

**Instalacja**  
python -m venv venv  
.\venv\Scripts\activate  

(Pobranie dependencies):  
pip install torch torchvision scikit-image numpy pillow  

**Trenowanie i wykorzystywanie modeli**  
Aby wytrenować nowy model należy uruchomić projekt train: python train.py  
Aby rozpocząć pracę z wytrenowanym modelem należy uruchomić gui: python gui.py  

**Konfiguracja treningu**  
NUM_EPOCHS  
Określa liczbę epok, czyli pełnych cykli przejścia przez cały zbiór danych treningowych.  

BATCH_SIZE  
Liczba obrazów przetwarzanych przez sieć w jednej iteracji.  

LEARNING_RATE   
Współczynnik uczenia, czyli "długość kroku" podczas optymalizacji.  

VALIDATION_SPLIT
Procent danych (od 0.0 do 1.0), który zostanie odłożony jako zbiór walidacyjny do testowania modelu po każdej epoce.  


