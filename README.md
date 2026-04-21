# Analyse comportementale de cachalots par Information Dirigée

Projet académique réalisé dans le cadre de Mission Sphyrna Odyssée.  
L'objectif est d'identifier un individu alpha au sein d'un groupe de six cachalots  
à partir de leurs trajectoires GPS, en utilisant la théorie de l'information dirigée.

---

##  Structure du projet
```
├── main.py                        # Script principal
├── Trajectoire_Cesar_sec.csv
├── Trajectoire_Fanny_sec.csv
├── Trajectoire_Marius_sec.csv
├── Trajectoire_Norine_sec.csv
├── Trajectoire_Felix_sec.csv
├── Trajectoire_Honore_sec.csv
```

---

##  Méthodologie

1. **Chargement** des trajectoires GPS brutes (longitude, latitude, profondeur, temps)
2. **Synchronisation** par ajustement polynomial (`fit_traj`) sur une grille temporelle commune
3. **Encodage** des déplacements en 8 états directionnels one-hot (`traj_to_W_onehot`)
4. **Calcul** de l'Information Dirigée (ID) pour chaque paire d'animaux
5. **Validation** via le test de Massey (ID(X→Y) + ID(Y→X) ≥ IM(X,Y))
6. **Analyse temporelle** par fenêtres glissantes (chevauchement 50%)
7. **Identification** de l'alpha par trois méthodes concordantes

---

## Résultats produits

- Matrice d'Information Dirigée (6×6)
- Score d'influence global par animal
- Évolution temporelle des scores par fenêtre glissante
- Identification de l'individu alpha

---

##  Dépendances
```bash
pip install numpy pandas matplotlib
```

---

##  Utilisation

Placer les fichiers CSV dans le même dossier que `main.py`, puis :
```bash
python main.py
```

---

##  Contexte théorique

L'**Information Dirigée** (Massey, 1990) mesure le flux d'information causal de X vers Y :

$$ID(X \to Y) = \sum_{i=1}^{n} \left[ H(Y_i \mid Y^{i-1}) - H(Y_i \mid Y^{i-1}, X^{i-1}) \right]$$

Contrairement à l'Information Mutuelle, elle est asymétrique et permet de détecter  
une relation de causalité directionnelle entre deux séries temporelles.

---
##  Données

Les trajectoires utilisées dans ce projet sont issues des campagnes d'observation de 
[Mission Sphyrna Odyssée](https://www.missionsphyrna.org), dédiée à l'étude et à la 
protection des cétacés en Méditerranée.

##  Auteur

Rapport rédigé par **Tonin PACHER**  
