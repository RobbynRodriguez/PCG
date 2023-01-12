Avant d’utiliser le code et afin de reproduire les résultats de la méthode il est nécessaire de télécharger l’archive all_patients.rar contenant les données du dataset “The China Physiological Signal Challenge 2020” sur ce drive (les données sont trop volumineuses pour qu’on puisse les mettre directement sur notre github) https://drive.google.com/drive/folders/1DoXdJJ5RCwPvahX9DSUTuWdltyK6iv0m
Une fois téléchargée, il suffit de décompresser l’archive et d’appeler le fichier test.py pour les ECG comme ci-dessous.


Le notebook s’utilise tel quel, quant au code de la méthode, il s’utilise en :
faisant ‘python3 test.py ECG’ afin de générer le fichier .mat non bruités pour le jeu de test utilisé par la méthode (“The China Physiological Signal Challenge 2020”).
‘python3 test.py PCG’ afin de générer le fichier .mat non bruités pour le jeu de test donné par le sujet (“ Signal Separation Evaluation Campaign 2016”).
‘python3 plot_ouputs.py ECG’ afin de générer les images résultats dans le dossier “demene_ECG”
‘python3 plot_ouputs.py PCG’ afin de générer les images résultats dans le dossier “demene_PCG”

Dépendances : 

Python3
Pytorch
Pytorch-Lightning
FastONN
Pandas
Numpy
Scipy
matplotlib
