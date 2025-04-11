
# Projet de Classification d'Images et Séries Temporelles par CNN et CNN-LSTM


## 📌 Description
Ce projet compare les performances des réseaux de neurones convolutifs (CNN) et des architectures hybrides CNN-LSTM pour deux tâches :
1. Classification d'images (dataset CIFAR-10)
2. Classification de séries temporelles (ECG5000 de l'archive UCR)

## 📦 Datasets Utilisés
| Dataset | Type | Taille | Classes |
|---------|------|--------|---------|
| [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) | Images | 60K (32x32) | 10 |
| [ECG5000](https://www.timeseriesclassification.com/dataset.php) | Séries temporelles | 5K (140 pts) | 5 |

## 🛠️ Installation
```bash
git clone https://github.com/LOUGBEGNON/classification_image.git
cd classification_image
```

### Configuration de l'environnement
Créez un environnement virtuel pour installer les dépendances :
```bash
python -m venv env
```

Activez l'environnement virtuel :
- Sur Windows :
  ```bash
  .\env\Scripts\activate
  ```
- Sur Unix ou MacOS :
  ```bash
  source env/bin/activate
  ```

### Installation des dépendances
Installez les dépendances du projet avec pip :
```bash
pip install -r requirements.txt
```

## 🧠 Architectures Implémentées

### CNN pour CIFAR-10
```python
Sequential(
  Conv2d(3→32, kernel=3), ReLU(), MaxPool2d(2),
  Conv2d(32→64, kernel=3), ReLU(), MaxPool2d(2),
  Conv2d(64→128, kernel=3), ReLU(), MaxPool2d(2),
  Flatten(),
  Linear(2048→512), Dropout(0.5),
  Linear(512→10)
)
```

### CNN-LSTM pour ECG5000
```python
Sequential(
  Conv1d(1→64, kernel=3), ReLU(), MaxPool1d(2),
  Conv1d(64→128, kernel=3), ReLU(), MaxPool1d(2),
  LSTM(128→64, bidirectional=True),
  Linear(128→5)
)
```

## 📊 Résultats

### CIFAR-10
| Modèle | Précision Train | Précision Test |
|--------|-----------------|----------------|
| CNN (sans aug) | 98% | 72% |
| CNN (avec aug) | 92% | 76% |

### ECG5000
| Modèle | Précision Train | Précision Test |
|--------|-----------------|----------------|
| CNN | 95% | 88% |
| CNN-LSTM | 93% | 91% |

## 🚀 Exécution
Il faut exécuter chaque fichier jupyter notebook.

## 📝 Conclusions
- Les CNN standards sont efficaces pour les images mais nécessitent de l'augmentation
- L'ajout de LSTM améliore les résultats sur les séries temporelles (+3%)
- Meilleure généralisation avec régularisation (Dropout)


## 👤 Auteur
- LOUGBEGNON G. AMEDEE - [@LOUGBEGNON](https://github.com/LOUGBEGNON)
- HABACK MARTHE Désirée Olivia
- DIALLO Ibrahima
