import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ===========================================================
# FONCTIONS INFORMATION
# ===========================================================

eps = 1e-12 #ajout pour eviter les division par 0 lorsque le résultat est 0

def Entropie(Z): #calcul entropie peut importe la dimension du cube 
    Z = np.array(Z, dtype=float) + eps
    Z = Z / Z.sum()
    Px = np.sum(Z, axis=0)
    return -np.sum(Px * np.log2(Px))

def Entropie_Y(Z): #calcul de l'entropie Y 
    Z = np.array(Z, dtype=float) + eps
    Z = Z / Z.sum()
    Py = np.sum(Z, axis=1)
    return -np.sum(Py * np.log2(Py))

def Entropie_jointe(Z): #calcul de l'entropie (X,Y)
    Z = np.array(Z, dtype=float) + eps
    P_Z = Z / np.sum(Z)
    return -np.sum(P_Z.flatten() * np.log2(P_Z.flatten()))

def Information_Mutuelle(Z): #calcul info mutuelle avec formule 
    return Entropie(Z) + Entropie_Y(Z) - Entropie_jointe(Z)

def entropie_marginale(Z): #exactement la même que l'entropie jointe 
    Z = np.array(Z, dtype=float).flatten() + eps   
    P = Z / Z.sum()
    return -np.sum(P * np.log2(P))

# ===========================================================
# INFO DIRIGÉE — codage one-hot binaire
# W shape : (2, T, 8) — vecteurs one-hot
# Pour chaque t : argmax du vecteur one-hot -> indice 0..7
# ID(X->Y) = H(Yt,Yp) + H(Xp,Yp) - H(Yp) - H(Yt,Xp,Yp)
# ===========================================================

def info_dirigee(W):
    n = 8
    T = W.shape[1]
    C = np.zeros((n, n, n)) + eps #evite les divisions par 0  
    for t in range(1, T): # à chaque pas de temps 
        yt = np.argmax(W[1, t,   :])   # present cible on observe si on reagi par rapport à l'action passé de la source 
        xp = np.argmax(W[0, t-1, :])   # passe source celui qui engendre l'action 
        yp = np.argmax(W[1, t-1, :])   # passe cible permet de controle si l'action était déjà engagé avant que la source agisse 
        C[yt, xp, yp] += 1 #cube final qui nous donne le nombre de fois ou la combinaison c'est réalisé 
    C_yt_yp = C.sum(axis=1) #on marginalise l'axe 0 sans se soucier de l'autre combien de fois y etait pareil au passé et au present 
    C_xp_yp = C.sum(axis=0) #on marginalise l'axe 1 sans se soucier de l'autre 
    C_yp    = C_yt_yp.sum(axis=0) #on récupère seulement le passé de Y
    id_score = (entropie_marginale(C_yt_yp) + entropie_marginale(C_xp_yp)- entropie_marginale(C_yp)  - entropie_marginale(C))
    return max(id_score, 0.0) #application de la formule i=1∑n​[H(Yi​∣Yi−1)−H(Yi​∣Yi−1,Xi)]


# ===========================================================
# CHARGEMENT DONNÉES
# ===========================================================

DOSSIER = os.path.dirname(os.path.abspath(__file__))

fichiers = {
    "Cesar":  "Trajectoire_Cesar_sec.csv",
    "Fanny":  "Trajectoire_Fanny_sec.csv",
    "Marius": "Trajectoire_Marius_sec.csv",
    "Norine": "Trajectoire_Norine_sec.csv",
    "Felix":  "Trajectoire_Felix_sec.csv",
    "Honore": "Trajectoire_Honore_sec.csv"
}
#recuperations des fichiers csv dans un dictionnaire 
data = {} #dictionnaire dans dictionnaire 
for nom, f in fichiers.items():
    chemin = os.path.join(DOSSIER, f) #construit le chemin vers les fichiers (pas obligatoire)
    try:
        df = pd.read_csv(chemin, header=None) #lecture des fichiers csv 
        data[nom] = { #permet de stocker seulement les clonnes qui nous interesse 
            'long': df[1].values, #definition longitude latitude à la colonne qui correspond du fichier et ajout dico 
            'lat':  df[2].values,
            'prof': df[3].values,
            'sec':  df[5].values
        }
        print(f"OK : {nom}") #confirmation de la lecture les fichiers 
    except Exception as e:
        print(f"Fichier manquant : {chemin} ({e})") #gestions des cas d'erreurs 

noms = list(data.keys()) #creation d'une liste de toutes les cle (nom des animaux) et ensuite stockage 
N = len(noms) #nombre d'animaux récupérés 
print(f"\n{N} animaux chargés : {noms}\n")

if N < 2:
    raise RuntimeError("Pas assez de fichiers chargés.") #en cas de problème de fichier 



# ===========================================================
# SYNCHRONISATION — temps normalisé [0,1] pour stabilité polyfit
# ===========================================================

t_min  = max(d['sec'].min() for d in data.values()) #pour chaque animal on recupère son départ en secondes puis on prend le max de ces departs 
t_max  = min(d['sec'].max() for d in data.values()) #pour chaque animal on rècupère son arrivée en secondes et on prend le min de ces arrivées 
#on possède donc maintenant les valeurs de depart et d'arrivée general 
t_sync = np.arange(t_min, t_max, 60) #cree une liste régulière avec saut de 60 de tmin à tmax 
t_norm = (t_sync - t_min) / (t_max - t_min) #normalise entre 0 et 1 les données [0 , 0.001 , ...., 1] pour empécher les bugs de polyfit 


for nom, d in data.items():
    print(f"{nom:8s} : début={d['sec'].min():.0f}s  fin={d['sec'].max():.0f}s  durée={(d['sec'].max()-d['sec'].min())/3600:.2f}h")


def fit_traj(nom, Ndeg, t_cible_norm, t0, t1):
    # permet de lisser les trajectoires et de les synchroniser
    # t0 et t1 sont les bornes de la fenêtre propre à chaque paire
    dat = data[nom]  # raccourci les commandes pour après
    t_base_norm = (dat['sec'] - t0) / (t1 - t0)  # normalisation sur les temps réels de l'animal
                                                   # irrégulier, propre à chaque paire (pas global)
    px = np.polyval(np.polyfit(t_base_norm, dat['long'], Ndeg), t_cible_norm)  # array régulier longitude
    py = np.polyval(np.polyfit(t_base_norm, dat['lat'],  Ndeg), t_cible_norm)  # pareil pour lat
    pz = np.polyval(np.polyfit(t_base_norm, dat['prof'], Ndeg), t_cible_norm)  # pareil pour prof
    return np.stack([px, py, pz], axis=1)  # crée la matrice des trois arrays après polyfit


def traj_to_W_onehot(traj_dict, noms_list): #creation du tenseur et traj   
    N_loc = len(noms_list) #compte le nombre d'animaux 
    T_loc = len(next(iter(traj_dict.values()))) - 1 #recupère toues les trajectoires du dictionnaire avec iter et next nous permet d'etre au premier element c'est a dire animal un et retourne un array long lat prof (matrice)
    W = np.zeros((N_loc, T_loc, 8), dtype=int) #crée le tenseur de l'animal rempli de zero 
    for i, nom in enumerate(noms_list): #parcour des animaux par indice 
        traj = traj_dict[nom] #recupère la trajectoire de l'animal courant 
        dx = np.diff(traj[:, 0])   #récupération Longitude
        dy = np.diff(traj[:, 1])   #récupération Latitude le signe indique le délpacement 
        dz = np.diff(traj[:, 2])   #récupération Profondeur
        angles = np.arctan2(dy, dx) #convertie en angle degres radian , comme une boussole 

        # Arrondi au cardinal le plus proche
        card = np.full(T_loc, -1, dtype=int)  # au cas ou un angle n'est pas assigné 
        card[(angles >= -np.pi/4) & (angles <   np.pi/4)]   = 0  # dans les quart de l'EST par exmple compris entre 0.78 et -0.78
        card[(angles >=  np.pi/4) & (angles <  3*np.pi/4)]  = 1  # dans le quart du Nord
        card[(angles <  -np.pi/4) & (angles >= -3*np.pi/4)] = 2  # dans le quart du Sud
        card[(angles >= 3*np.pi/4) | (angles <  -3*np.pi/4)] = 3  # dans le quart de l'Ouest
        assert np.all(card >= 0), f"Angle non couvert détecté pour {nom} : indices {np.where(card < 0)}" #s'assurer que les cases sont toues indexé 

        montee = (dz < 0).astype(int)   #on converties les + et - en booleen qu'on re transforme en entier donc 0 et 1
        etat   = card * 2 + montee      # indice 0..7
        # Encodage one-hot : 1 à la position etat, 0 ailleurs
        for t in range(T_loc): # a cahque instant t on ajoute etat au tenseur donc le 1 à la position voulue 
            W[i, t, etat[t]] = 1 #tout ceux pair descende et impair monte 
    return W #tenseur terminé 


# ===========================================================
# N FIXÉ À 8
# ===========================================================

N_optimal = 9 #valeur de decoupage du cube 
print(f"N fixé à {N_optimal}\n") #affichage de la fixation 


# ===========================================================
# TRAJECTOIRES ET TENSEUR W FINAL (one-hot)
# ===========================================================

traj_final = {nom: fit_traj(nom, N_optimal, t_norm, t_min, t_max) for nom in noms} #stockage des données lissé dans le dictionnaire avec appel de la fonction fittraj
W_final    = traj_to_W_onehot(traj_final, noms)   # shape (N, T, 8) #creation du tenseur avec les données lissé (MAJORITAIREMENT INUTILE A PART POUR LE GLISSEMENT)
T          = W_final.shape[1] #retourne le nombre d'instant T dans le tenseur 
print(f"Tenseur W_final : shape = {W_final.shape}  →  {W_final.shape[0]} animaux × {W_final.shape[1]} instants × {W_final.shape[2]} états")

print("\nTemps d'émission par animal :")
for nom in noms:
    d = data[nom]
    debut  = max(d['sec'].min(), t_min)
    fin    = min(d['sec'].max(), t_max)
    duree  = (fin - debut) / 3600
    n_pts  = len(d['sec'])
    print(f"  {nom:8s} : {debut:.0f}s → {fin:.0f}s  |  durée = {duree:.2f}h  |  {n_pts} points GPS bruts")

print(W_final)

# ===========================================================
# MATRICE INFORMATION DIRIGÉE GLOBALE
# ===========================================================

mat_global = np.zeros((N, N)) #matrice remplie de 0 avec une case par paire 
for i in range(N):
    for j in range(N): #parcours toutes les combinaisons possibles des deux animaux 
        if i == j: #si on teste le meme animal avec lui meme on zap 
            continue
        t0 = max(data[noms[i]]['sec'].min(), data[noms[j]]['sec'].min())  # debut commun à la paire
        t1 = min(data[noms[i]]['sec'].max(), data[noms[j]]['sec'].max())  # fin commune à la paire
        t_norm_ij = (np.arange(t0, t1, 60) - t0) / (t1 - t0)            # grille normalisée propre à la paire donc array entre 0 et 1
        traj_i = fit_traj(noms[i], N_optimal, t_norm_ij, t0, t1) #synchronisation des deux avec fit_traj
        traj_j = fit_traj(noms[j], N_optimal, t_norm_ij, t0, t1)
        W_ij   = traj_to_W_onehot({noms[i]: traj_i, noms[j]: traj_j}, [noms[i], noms[j]]) #creation du tenseur de i et j avec les données normalisé et lissé 
        mat_global[i, j] = info_dirigee(W_ij) #calcul de l'info dirigé entre les deux animaux 
scores_global = mat_global.sum(axis=1) #somme chaque ligne de la matrice pour avoir l'info dirigé total de chaque animal 
alpha_global  = noms[np.argmax(scores_global)] #trouve l'indice le plus eleve donc l'alpha 

print("MATRICE ID (ligne = source, colonne = cible)")
print(pd.DataFrame(mat_global, index=noms, columns=noms).round(3))
# ===========================================================
# MATRICE COOCCURRENCE — factorisée pour éviter la duplication
# ===========================================================

def cooccurrence(W, i, j, n=8):
    """Matrice de cooccurrence des états (argmax one-hot) entre animaux i et j."""
    Z = np.zeros((n, n)) + eps #cree une matrice 8x8 vide 
    for t in range(W.shape[1]): #parcours tou les instant 
        Z[np.argmax(W[i, t]), np.argmax(W[j, t])] += 1 #recupère l'etat de l'animal i et j a l'instant t 
    return Z #cree la matrice des deplacements et donc compare si les deplacement sont lié 


# ===========================================================
# TEST MASSEY — résidu = ID_AB + ID_BA - IM >= 0
# ===========================================================

print("\nTEST MASSEY") #teste de masset pour chercher si les resultats sont juste 
for i in range(N):
    for j in range(i + 1, N): #parcour seulement les paires une seule foid (penser à l'algo moyenne tri insertion)
        idij   = mat_global[i, j] #recupère l'ID calculé dans le sens 1
        idji   = mat_global[j, i] #recupère l'ID calculé dans le sens 2
        t0 = max(data[noms[i]]['sec'].min(), data[noms[j]]['sec'].min()) #recupère la fenetre comune 
        t1 = min(data[noms[i]]['sec'].max(), data[noms[j]]['sec'].max())
        t_norm_ij = (np.arange(t0, t1, 60) - t0) / (t1 - t0) # pareil qu'avant on normalise le array 
        traj_i = fit_traj(noms[i], N_optimal, t_norm_ij, t0, t1) #lissage des trajectoires  pour i et j
        traj_j = fit_traj(noms[j], N_optimal, t_norm_ij, t0, t1)
        W_ij   = traj_to_W_onehot({noms[i]: traj_i, noms[j]: traj_j}, [noms[i], noms[j]]) #creation du tenseur avec les deux animaux
        Z      = cooccurrence(W_ij, 0, 1) #creation de la matrice de deplacement W_ij
        im     = Information_Mutuelle(Z) #calcul de l'infomut de Z
        residu = idij + idji - im #on neglige le -1 car avec des pas de 60s il n'y a pas de causalité avec le passé 
        flag   = "OK" if residu >= -1e-6 else "ATTENTION negatif" #affichage 
        print(f"  {noms[i]} <-> {noms[j]} : résidu = {residu:.4f}  [{flag}]")


# ===========================================================
# FENÊTRES GLISSANTES 50% #mise en place des fenetres glissantes
# ===========================================================

TAILLE_MIN_FENETRE = 10   # minimum absolu (adaptatif selon T)
taille_f = max(T // 2, TAILLE_MIN_FENETRE)
pas_f    = taille_f // 2
debuts_f = list(range(0, T - taille_f + 1, pas_f))

history_alpha  = []
history_scores = []

print(f"\nFENÊTRES GLISSANTES (chevauchement 50%)")
print(f"  Taille fenêtre : {taille_f} pas | Pas : {pas_f} | Nb fenêtres : {len(debuts_f)}")

if not debuts_f:
    print("  ATTENTION : T trop petit pour découper en fenêtres — analyse globale uniquement.")
else:
    for d in debuts_f:
        Wf = W_final[:, d:d + taille_f, :]   # shape (N, taille_f, 8)
        sc = [sum(info_dirigee(Wf[[i, j]]) for j in range(N) if j != i) for i in range(N)]
        leader = noms[np.argmax(sc)]
        history_alpha.append(leader)
        history_scores.append(sc)
        print(f"  [{d:4d}:{d+taille_f:4d}] -> leader : {leader:8s}  scores : {[round(x,3) for x in sc]}")


# ===========================================================
# ALPHA FINAL — 3 méthodes concordantes test general pas important 
# ===========================================================

counts         = {n: history_alpha.count(n) for n in noms} #resume des fenetres glissantes
alpha_temporel = max(counts, key=counts.get) if history_alpha else alpha_global
scores_cumul   = np.sum(history_scores, axis=0) if history_scores else scores_global
alpha_cumul    = noms[np.argmax(scores_cumul)]

print("\n============================")
print("RÉSULTAT FINAL")
print("============================")
print(f"Alpha global          : {alpha_global}")
print(f"Alpha temporel (vote) : {alpha_temporel}  {counts}")
print(f"Alpha cumulé          : {alpha_cumul}")

candidats = [alpha_global, alpha_temporel, alpha_cumul]
if len(set(candidats)) == 1:
    print(f"\n>>> ALPHA FINAL FIABLE : {alpha_global} (3/3 méthodes concordent) <<<")
elif candidats.count(alpha_temporel) >= 2:
    print(f"\n>>> ALPHA PROBABLE : {alpha_temporel} (2/3 méthodes concordent) <<<")
else:
    print(f"\n>>> RÉSULTAT AMBIGU — vérifier manuellement <<<")
    print(f"    Global={alpha_global} | Temporel={alpha_temporel} | Cumulé={alpha_cumul}")

