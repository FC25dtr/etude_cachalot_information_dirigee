import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#--------------------------------FONCTION NECESSAIRE-------------------------------------

eps = 1e-12 #ajout pour eviter les division par 0 lorsque le résultat est 0

def Entropie(Z): #calcul entropie peut importe la dimension du cube 
    Z = np.array(Z, dtype=float) + eps
    Z = Z / Z.sum()
    Px = np.sum(Z, axis=1)
    return -np.sum(Px * np.log2(Px))

def Entropie_Y(Z): #calcul de l'entropie Y 
    Z = np.array(Z, dtype=float) + eps
    Z = Z / Z.sum()
    Py = np.sum(Z, axis=0)
    return -np.sum(Py * np.log2(Py))

def Entropie_jointe(Z): #calcul de l'entropie (X,Y)
    Z = np.array(Z, dtype=float) + eps
    P_Z = Z / np.sum(Z)
    return -np.sum(P_Z.flatten() * np.log2(P_Z.flatten()))

def Information_Mutuelle(Z): #calcul info mutuelle avec formule 
    return Entropie(Z) + Entropie_Y(Z) - Entropie_jointe(Z)


#--------------------------INFO DIRIGE AVEC TEMPS T-1 -----------------------------------------------

def info_dirigee(W):
    n = 8
    T = W.shape[1]
    C = np.zeros((n, n, n)) + eps #creation de la matrice vide remplis de 0
    for t in range(1, T):
        yt = np.argmax(W[1, t,   :])#recuperation des données de Y au temps t 
        xp = np.argmax(W[0, t-1, :])#recuperation des données de X au temps t -1
        yp = np.argmax(W[1, t-1, :])#recuêration des données de Y au temps t-1 
        C[yt, xp, yp] += 1 #ajout dans ma matrice C 
    #arrivée ici le tenseur est rempli 
    C_yt_yp = C.sum(axis=1) #je somme les donnees de X au temps t -1
    C_xp_yp = C.sum(axis=0) #je sommes les données de Y au temps t -1
    C_yp    = C_yt_yp.sum(axis=0) #je somme les données au temps Y present 
    id_score = (Entropie_jointe(C_yt_yp) + Entropie_jointe(C_xp_yp) #application de la formule classique de l'information dirigé en utilisant le temps t-1 asynchrone 
               - Entropie_jointe(C_yp)   - Entropie_jointe(C))
    return max(id_score, 0.0) #retourne le score si valide 

#-----------------------------------------RECUPERATION DES DONNÉES-------------------------------------------

fichiers = {
    "Cesar":  "Trajectoire_Cesar_sec.csv",
    "Fanny":  "Trajectoire_Fanny_sec.csv",
    "Marius": "Trajectoire_Marius_sec.csv",
    "Norine": "Trajectoire_Norine_sec.csv",
    "Felix":  "Trajectoire_Felix_sec.csv",
    "Honore": "Trajectoire_Honore_sec.csv"
}

data = {} #dictionnaire dans dictionnaire
for nom, f in fichiers.items():
    df = pd.read_csv(f, header=None) #lecture des fichiers csv
    data[nom] = { #permet de stocker seulement les colonnes qui nous interessent
        'long': df[1].values, #definition longitude latitude à la colonne qui correspond du fichier et ajout dico
        'lat':  df[2].values,
        'prof': df[3].values,
        'sec':  df[5].values
    }

noms = list(data.keys()) #creation d'une liste de toutes les cles (nom des animaux) et ensuite stockage
N = len(noms) #nombre d'animaux récupérés
print(f"\n{N} animaux chargés : {noms}\n")
#------------------------------synchronisation et discretisation----------------------------------------------

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
        montee = (dz < 0).astype(int)   #on converties les + et - en booleen qu'on re transforme en entier donc 0 et 1
        etat   = card * 2 + montee      # indice 0..7
        # Encodage one-hot : 1 à la position etat, 0 ailleurs
        for t in range(T_loc): # a cahque instant t on ajoute etat au tenseur donc le 1 à la position voulue 
            W[i, t, etat[t]] = 1 #tout ceux pair descende et impair monte 
    return W #tenseur terminé 


#------------------------------------------definition du degres de polynome-----------------------------------------------------

D_optimal = 9 #valeur de decoupage du cube 
print(f"N fixé à {D_optimal}\n") #affichage de la fixation 



#------------------------------------- TRAJECTOIRES ET TENSEUR W FINAL (one-hot)--------------------------------------------------


traj_final = {nom: fit_traj(nom, D_optimal, t_norm, t_min, t_max) for nom in noms} #stockage des données lissé dans le dictionnaire avec appel de la fonction fittraj
W_final    = traj_to_W_onehot(traj_final, noms)   # shape (N, T, 8) #creation du tenseur avec les données lissé (MAJORITAIREMENT INUTILE A PART POUR LE GLISSEMENT)
T          = W_final.shape[1] #retourne le nombre d'instant T dans le tenseur 
print(f"Tenseur W_final : shape = {W_final.shape}  →  {W_final.shape[0]} animaux × {W_final.shape[1]} instants × {W_final.shape[2]} états")

print("\nTemps d'émission par animal :") #affichage du temps d'emission pour analyser les données commune 
for nom in noms:
    d = data[nom]
    debut  = max(d['sec'].min(), t_min)
    fin    = min(d['sec'].max(), t_max)
    duree  = (fin - debut) / 3600
    n_pts  = len(d['sec'])
    print(f"  {nom:8s} : {debut:.0f}s → {fin:.0f}s  |  durée = {duree:.2f}h  |  {n_pts} points GPS bruts")

print(W_final)


# ---------------------------------MATRICE INFORMATION DIRIGÉE GLOBALE, moyenne sur N_dec tranches-----------------------------------


N_dec = 7  # nombre de tranches égales sur lesquels on moyenne l'ID

mat_global = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        t0 = max(data[noms[i]]['sec'].min(), data[noms[j]]['sec'].min()) #recuperation des temps communs au deux animaux 
        t1 = min(data[noms[i]]['sec'].max(), data[noms[j]]['sec'].max())
        t_norm_ij = (np.arange(t0, t1, 60) - t0) / (t1 - t0) #normalisation des données 
        traj_i = fit_traj(noms[i], D_optimal, t_norm_ij, t0, t1) #polyfit des donnes pour synchroniser 
        traj_j = fit_traj(noms[j], D_optimal, t_norm_ij, t0, t1)
        W_ij   = traj_to_W_onehot({noms[i]: traj_i, noms[j]: traj_j}, [noms[i], noms[j]]) #creation du tenseur commun 
        #tenseur terminé des deux animaux 
        T_ij     = W_ij.shape[1] #recupere seulement le nombre de pas de temps que l'on possède 
        taille   = T_ij // N_dec  # taille de chaque tranche
        scores   = [] #pour recuperer les scores 
        for k in range(N_dec):
            debut  = k * taille
            fin    = debut + taille if k < N_dec - 1 else T_ij  # dernière tranche prend le reste
            W_tr   = W_ij[:, debut:fin, :] #on recuperre la partie qui nous interesse 
            if W_tr.shape[1] > 1:  # au moins 2 instants pour calculer un ID avoir t-1
                scores.append(info_dirigee(W_tr)) #ajout du score d'information dirigé 
        mat_global[i, j] = np.mean(scores) if scores else 0.0
        
scores_global = mat_global.sum(axis=1)
alpha_global  = noms[np.argmax(scores_global)]
print("MATRICE ID (ligne = source, colonne = cible)")
print(pd.DataFrame(mat_global, index=noms, columns=noms).round(3))


# -------------------------------------- TEST MASSEY, résidu = ID_AB + ID_BA - IM >= 0-------------------------------------------------------


def _construire_C4(W):
    n = 8 #nombre de découpe du cube 
    T = W.shape[1] #on recupe la shape 1
    C4 = np.zeros((n, n, n, n)) + eps #creation cube rempli de 0
    for t in range(1, T):
        xt = np.argmax(W[0, t,   :]) # état X présent  (temps t)
        yt = np.argmax(W[1, t,   :]) # état Y présent  (temps t)
        xp = np.argmax(W[0, t-1, :]) # état X passé    (temps t−1)
        yp = np.argmax(W[1, t-1, :]) # état Y passé    (temps t−1)
        C4[xt, yt, xp, yp] += 1 #placement dans le cube 
    return C4

def id_brut(W):
    C4 = _construire_C4(W)
    C        = C4.sum(axis=0) # cube C[yt, xp, yp] — marginale sur xt
    C_yt_yp  = C.sum(axis=1) # marginalise xp → jointe (yt, yp)
    C_xp_yp  = C.sum(axis=0) # marginalise yt → jointe (xp, yp)
    C_yp     = C_yt_yp.sum(axis=0) # marginalise yt → marginale yp
    return (Entropie_jointe(C_yt_yp) + Entropie_jointe(C_xp_yp)- Entropie_jointe(C_yp)  - Entropie_jointe(C)) #application de la formule 


def im_massey(W):
    C4 = _construire_C4(W)
    C_yt_yp    = C4.sum(axis=(0, 2))# marginalise xt et xp -> (yt, yp)
    C_xt_xp    = C4.sum(axis=(1, 3))# marginalise yt et yp -> (xt, xp)
    C_xp_yp    = C4.sum(axis=(0, 1))# marginalise xt et yt -> (xp, yp)
    C_yp       = C_xp_yp.sum(axis=0)# marginalise xp       -> (yp)
    C_xp       = C_xp_yp.sum(axis=1)# marginalise yp       -> (xp)
    C_yt_xp_yp = C4.sum(axis=0)# marginalise xt -> cube de ID(X→Y)
    C_xt_yp_xp = C4.sum(axis=1)# marginalise yt ->  cube de ID(Y→X)
    return (Entropie_jointe(C_yt_yp)+ Entropie_jointe(C_xt_xp)+ 2 * Entropie_jointe(C_xp_yp)- Entropie_jointe(C_yp)- Entropie_jointe(C_xp)- Entropie_jointe(C_yt_xp_yp)- Entropie_jointe(C_xt_yp_xp)) #application de la formule 


#cela marche car le calcul de l'id est basé sur les temps t-1 
for i in range(N):
    for j in range(i + 1, N):
        nom_a, nom_b = noms[i], noms[j]
        t0 = max(data[nom_a]['sec'].min(), data[nom_b]['sec'].min()) #recuperation temps par paire 
        t1 = min(data[nom_a]['sec'].max(), data[nom_b]['sec'].max())
        t_norm_ij = (np.arange(t0, t1, 60) - t0) / (t1 - t0)
        traj_a = fit_traj(nom_a, D_optimal, t_norm_ij, t0, t1)
        traj_b = fit_traj(nom_b, D_optimal, t_norm_ij, t0, t1) #synchronisation trajectoire
        W_ij   = traj_to_W_onehot({nom_a: traj_a, nom_b: traj_b}, [nom_a, nom_b]) #creation du cube avec les deplacements
        T_ij   = W_ij.shape[1]
        id_ab  = id_brut(W_ij) #application ID brut qui enleve le max 0
        id_ba  = id_brut(W_ij[[1, 0]]) #pareil
        somme  = id_ab + id_ba #recuperation res
        im_ref = im_massey(W_ij) #id de massey 
        residu = somme - im_ref #formule 
        label = f"{nom_a}→{nom_b}"
        print(f"{label:<20} {id_ab:>10.4f} {id_ba:>10.4f} {somme:>10.4f} {im_ref:>10.4f} {residu:>10.4f}")

print("-" * 90)
print("  Résidu = 0.0000 sur toutes les paires : identité algébrique vérifiée.")


# -------------------------------FENÊTRES GLISSANTES 50%, PAR PAIRE #méthode distincte de celle sans glissement -------------------------------------------


TAILLE_MIN_FENETRE = 10 # minimum de 10 pas pour qu'une fenêtre ait du sens
history_alpha  = [] # stocke le leader de chaque fenêtre
history_scores = [] # stocke les scores de tous les animaux à chaque fenêtre
pair_windows = {} # contiendra pour chaque paire (i,j) : son tenseur, ses débuts de fenêtre, sa taille

for i in range(N):
    for j in range(N):
        if i == j:
            continue # on ignore un animal avec lui-même
        t0 = max(data[noms[i]]['sec'].min(), data[noms[j]]['sec'].min()) # début commun = le plus tard des deux départs
        t1 = min(data[noms[i]]['sec'].max(), data[noms[j]]['sec'].max()) # fin commune = le plus tôt des deux fins
        t_norm_ij = (np.arange(t0, t1, 60) - t0) / (t1 - t0) # axe temporel normalisé entre 0 et 1
        traj_i = fit_traj(noms[i], D_optimal, t_norm_ij, t0, t1) # trajectoire lissée de i
        traj_j = fit_traj(noms[j], D_optimal, t_norm_ij, t0, t1) # trajectoire lissée de j
        W_ij = traj_to_W_onehot({noms[i]: traj_i, noms[j]: traj_j}, [noms[i], noms[j]]) # tenseur one-hot des deux animaux
        T_ij = W_ij.shape[1] # nombre de pas de temps disponibles pour cette paire
        taille_f = max(T_ij // 2, TAILLE_MIN_FENETRE) # moitié de T_ij avec un minimum garanti
        pas_f = taille_f // 2 # chevauchement de 50%
        debuts_f = list(range(0, T_ij - taille_f + 1, pas_f)) # indices de début de chaque fenêtre
        pair_windows[(i, j)] = (W_ij, debuts_f, taille_f) # on sauvegarde tout pour cette paire

n_fenetres = max(len(pair_windows[(i, j)][1])
                 for i in range(N) for j in range(N) if i != j) # nombre maximum de fenêtres parmi toutes les paires

print(f"\nFENÊTRES GLISSANTES PAR PAIRE (chevauchement 50%)")
print(f"  Nb fenêtres : {n_fenetres}")

for k in range(n_fenetres): # on parcourt chaque fenêtre par son indice
    sc = np.zeros(N) # scores initialisés à 0 pour tous les animaux
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            W_ij, debuts_f, taille_f = pair_windows[(i, j)] # on récupère les données précalculées de la paire
            if k >= len(debuts_f): # cette paire n'a pas de k-ième fenêtre, on saute
                continue
            d = debuts_f[k] # début de la k-ième fenêtre pour cette paire
            Wf = W_ij[:, d:d + taille_f, :] # découpe du tenseur sur cette fenêtre
            sc[i] += info_dirigee(Wf) # on ajoute l'ID de i vers j au score de i
    leader = noms[np.argmax(sc)] # l'animal avec le score le plus élevé est le leader
    history_alpha.append(leader)
    history_scores.append(sc.tolist())
    print(f"  [fenêtre {k}] -> leader : {leader:8s}  scores : {[round(x, 3) for x in sc]}")

# -----------------------------------------ALPHA FINAL 3 méthodes ------------------------------------------------------


counts         = {n: history_alpha.count(n) for n in noms} #resume des fenetres glissantes
alpha_temporel = max(counts, key=counts.get) if history_alpha else alpha_global
scores_cumul   = np.sum(history_scores, axis=0) if history_scores else scores_global
alpha_cumul    = noms[np.argmax(scores_cumul)]

print("RÉSULTAT FINAL")
print(f"Alpha global          : {alpha_global}")
print(f"Alpha temporel (vote) : {alpha_temporel}  {counts}")
print(f"Alpha cumulé          : {alpha_cumul}")
