import os
import scipy.io.wavfile
import numpy as np
import base
import matplotlib.pyplot as plt
import random
from sklearn import decomposition, cluster, metrics, neural_network, svm, ensemble
# Notez le code de preuve de dépôt : "800d0f2def176fe1cb5529a227f6cc55"
Prefixe=['aa','ee','eh','eu','ii','oe','oh','oo','uu','yy']

#Variables à modifier pour effectuer différents tests
pourcentage_test = 90    #10% des fichiers seront des tests, modification possible (max = 98)
n = 7    #reduction de la dimension PCA, modification possible (max 13)

# Sélection des bases apprentissages et tests
 
liste_nom_fichier_apprentissage = []
liste_nom_fichier_test = []

liste_fichiers = os.listdir("Signaux/")
for NomFichier in liste_fichiers:
    if(int(NomFichier[2:4])<pourcentage_test):
        liste_nom_fichier_test.append(NomFichier)
    else:
        liste_nom_fichier_apprentissage.append(NomFichier)

print()
print("Nombre fichiers pour apprentissage : ", len(liste_nom_fichier_apprentissage))
print("Nombre fichiers pour test : ", len(liste_nom_fichier_test))

# Analyse du signal (fichiers apprentissage)

liste_fe_echant = []
liste_classe = []
liste_dsp = []
matrice = []

for NomFichier in liste_nom_fichier_apprentissage:
    (Fe,Echantillons)=scipy.io.wavfile.read("Signaux/"+NomFichier)
    liste_fe_echant.append((Fe,Echantillons))
    NumerClasse=Prefixe.index(NomFichier[0:2])
    liste_classe.append(NumerClasse)
    dsp=np.abs(np.fft.fft(Echantillons))
    liste_dsp.append(dsp)
    VecteurCoefficients = base.mfcc(Echantillons, samplerate=Fe, winlen=(len(Echantillons)/Fe), winstep=(len(Echantillons)/Fe),nfft=1024)
    matrice.append(VecteurCoefficients[0])

# Affichage représentation temporelle et frequentielle (3 fichiers aléatoires)

nb1 = random.randint(0,len(liste_fe_echant))
nb2 = random.randint(0,len(liste_fe_echant))
nb3 = random.randint(0,len(liste_fe_echant))
plt.subplot(321)
plt.plot(liste_fe_echant[nb1][1],"r")
plt.title("Représentation temporelle de " + Prefixe[liste_classe[nb1]])
plt.subplot(322)
plt.plot(liste_dsp[nb1],"r")
plt.title("Représentation frequentielle de " + Prefixe[liste_classe[nb1]])
plt.subplot(323)
plt.plot(liste_fe_echant[nb2][1],"b")
plt.title("Représentation temporelle de " + Prefixe[liste_classe[nb2]])
plt.subplot(324)
plt.plot(liste_dsp[nb2],"b")
plt.title("Représentation frequentielle de " + Prefixe[liste_classe[nb2]])
plt.subplot(325)
plt.plot(liste_fe_echant[nb3][1],"y")
plt.title("Représentation temporelle de " + Prefixe[liste_classe[nb3]])
plt.subplot(326)
plt.plot(liste_dsp[nb3],"y")
plt.title("Représentation frequentielle de " + Prefixe[liste_classe[nb3]])

plt.subplots_adjust(hspace = 0.9, wspace = 0.45)
#plt.show()

# resultats
matrice_coef = np.array(matrice)


# Analyse du signal (fichiers test)
matrice_coef_test = []
liste_classe_test = []

for NomFichier in liste_nom_fichier_test:
    (Fe,Echantillons)=scipy.io.wavfile.read("Signaux/"+NomFichier)
    NumerClasse=Prefixe.index(NomFichier[0:2])
    liste_classe_test.append(NumerClasse)
    VecteurCoefficients = base.mfcc(Echantillons, samplerate=Fe, winlen=(len(Echantillons)/Fe), winstep=(len(Echantillons)/Fe),nfft=1024)
    matrice_coef_test.append(VecteurCoefficients[0])

matrice_coef_test = np.array(matrice_coef_test)

# Prétraitement des données :
#https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

print("n_components pca = ", n)
print()
pca = decomposition.PCA(n_components=n)
pca.fit(matrice_coef)
matrice_coef = pca.transform(matrice_coef)
matrice_coef_test = pca.transform(matrice_coef_test)

# Algorithme des centres mobiles
print("Algorithme des centres mobiles")
classif=cluster.KMeans(len(Prefixe))
classif.fit(matrice_coef)
y_predict=classif.predict(matrice_coef_test)
#print(y_predict)
reussite = metrics.confusion_matrix(liste_classe_test, y_predict)
taux = 100
for i in range(10):
    aux = 0
    for j in range(10):
        if reussite[j][i]>aux:
            aux = reussite[j][i]
    if aux<pourcentage_test:
        taux += aux-pourcentage_test
print("Bonnes réponses (environ) : ", "%.2f" % taux, "%")
print()

# Perceptron multicouche
print("Perceptron multicouche")
classif = neural_network.MLPClassifier()
classif.fit(matrice_coef,liste_classe)
y_predict=classif.predict(matrice_coef_test)
#print(y_predict)
reussite = metrics.confusion_matrix(liste_classe_test, y_predict)
taux = 0
for i in range(10):
    taux += reussite[i][i]
taux = taux*100/(10*pourcentage_test)
print("Bonnes réponses : ", "%.2f" % taux, "%")
print()

# Separateur à Vaste Marge
print("Separateur à Vaste Marge")
classif = svm.SVC()
classif.fit(matrice_coef,liste_classe)
y_predict=classif.predict(matrice_coef_test)
#print(y_predict)
reussite = metrics.confusion_matrix(liste_classe_test, y_predict)
taux = 0
for i in range(10):
    taux += reussite[i][i]
taux = taux*100/(10*pourcentage_test)
print("Bonnes réponses : ", "%.2f" % taux, "%")
print()

# Forêts d’arbres décisionnels :
print("Forêts d’arbres décisionnels")
classif = ensemble.RandomForestClassifier()
classif.fit(matrice_coef,liste_classe)
y_predict=classif.predict(matrice_coef_test)
#print(y_predict)
reussite = metrics.confusion_matrix(liste_classe_test, y_predict)
taux = 0
for i in range(10):
    taux += reussite[i][i]
taux = taux*100/(10*pourcentage_test)
print("Bonnes réponses : ", "%.2f" % taux, "%")
print()