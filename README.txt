# Astralis 🚀 — Moteur de Simulation Orbitale N-Corps 3D

Astralis est un logiciel de simulation gravitationnelle 3D doté d'une interface graphique d'analyse complète. Il permet de modéliser la dynamique de systèmes célestes complexes en intégrant des calculs physiques avancés.

## 📝 Spécifications & Conception
* **Architecture et physique** : L. Capdevila
* **Développement du code** : Claude (IA) & Gemini (IA)

## 🌟 Fonctionnalités physiques
* **Intégrateur Symplectique de Yoshida (4ème ordre)** : Conservation de l'énergie (Hamiltonien) sur de très longues durées.
* **Pas de temps adaptatif (Sundman / Cinématique)** : Précision maximale lors des rencontres proches.
* **Perturbations avancées** : Prise en compte de l'aplatissement des corps ($J_2$), des couples de marée ($k_2$ de Love), et de la perte ou du gain de masse ($\dot{M}$).
* **Corrections relativistes** : Premier ordre post-newtonien (PN1) pour la précession des périhélies.

## 📊 Tableau de bord d'analyse (PyQt5)
* Visualisation des orbites en 2D et 3D temps réel (Matplotlib & Three.js).
* Analyse des distances inter-corps et histogrammes de répartition.
* Suivi de la dérive d'énergie et du moment cinétique.
* Analyse de la stabilité via les éléments orbitaux de Kepler ($e$, $a$, $i$, $\omega$, $\Omega$).
* Cartes de densité spatiale et diagrammes de phase.

## 🛠️ Installation & Lancement

### Version installable (Windows)
Vous pouvez télécharger l'installateur autonome (`Setup_Astralis.exe`) directement dans l'onglet **Releases** à droite de cette page. Aucune installation de Python n'est requise.

### Lancement depuis les sources
1. Clonez le dépôt :
   ```bash
   git clone https://github.com/L-Capdevila/Astralis.git