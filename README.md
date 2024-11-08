# Video Auto Rating

Ce projet est un pipeline automatisé de traitement de vidéos. Il utilise plusieurs bibliothèques pour compresser des vidéos, extraire les transcriptions, analyser les gestes et le ton de la voix, et finalement générer des scores d'évaluation pour chaque vidéo. Les résultats sont exportés dans un fichier CSV pour faciliter le suivi et l'analyse.

## Fonctionnalités

1. **Compression de vidéos** : Les vidéos sont compressées en utilisant FFmpeg pour réduire leur taille tout en maintenant une qualité optimale.
2. **Extraction des noms et prénoms** : Le script identifie automatiquement le nom, le prénom et le groupe de chaque étudiant à partir du nom de fichier (ex. : `NOM_Prénom_Groupe.mp4`).
3. **Transcription audio** : Utilisation du modèle Whisper pour transcrire les fichiers vidéo.
4. **Analyse non-verbale** : Analyse des mouvements et de la posture des étudiants en utilisant MediaPipe.
5. **Calcul du débit de parole et du ton** : Extraction des caractéristiques audio pour calculer le débit de parole et la variation du ton.
6. **Évaluation de la présentation avec OpenAI** : Utilisation de l'API OpenAI pour évaluer la transcription en fonction de critères d'évaluation précis.
7. **Export des résultats** : Les scores et informations pertinentes (nom, prénom, groupe, question posée, compétence) sont exportés dans un fichier CSV pour un suivi détaillé.

## Prérequis

- **Python 3.10+**
- **Bibliothèques Python** :
  - `whisper`
  - `pandas`
  - `numpy`
  - `mediapipe`
  - `opencv-python`
  - `librosa`
  - `absl-py`
  - `openai`
  - `python-dotenv`

Pour installer toutes les dépendances, exécutez :
```bash
pip install -r requirements.txt
```

## Configuration

### Clé API OpenAI

1. Créez un fichier `.env` à la racine du projet.
2. Ajoutez votre clé API OpenAI dans le fichier `.env` comme suit :
   ```plaintext
   OPENAI_API_KEY=YOUR_OPENAI_API_KEY
   ```

## Utilisation

Pour exécuter le script, lancez la commande suivante :

```bash
python traitement_videos.py
```


## Le script effectue les étapes suivantes :

1. **Compression des vidéos** : Compresse les vidéos présentes dans le dossier `data/videos` et stocke les versions compressées dans `data/videos_compressed`.
2. **Extraction des noms, prénoms et groupes** : Extrait automatiquement les informations des étudiants (nom, prénom, groupe) à partir des noms de fichiers.
3. **Transcription avec Whisper** : Transcrit le contenu audio des vidéos en utilisant le modèle Whisper.
4. **Analyse des gestes non-verbaux** : Utilise MediaPipe pour analyser les mouvements et gestes non-verbaux dans les vidéos.
5. **Calcul du débit de parole et du ton** : Calcule la vitesse de parole et les variations de ton à l'aide de `librosa`.
6. **Évaluation de la transcription avec OpenAI** : Identifie la question posée par l'étudiant et la compétence associée dans la transcription, grâce à l'API d'OpenAI.
7. **Export des résultats** : Génère un fichier CSV `resultats_evaluation.csv` contenant les scores et informations de chaque vidéo.

## Exemple de log

Pendant l'exécution, vous verrez des messages de progression similaires à ceux-ci :

```yaml
2024-11-08 22:07:30,406 - INFO - Début de la compression des vidéos.
2024-11-08 22:07:30,406 - INFO - Vidéo déjà compressée : data/videos_compressed/BETHOUX_camille.mp4
2024-11-08 22:07:30,406 - INFO - Compression des vidéos terminée ✅.
2024-11-08 22:07:30,408 - INFO - Extraction des noms et prénoms terminée.
...
2024-11-08 22:11:06,075 - INFO - DataFrame exporté en fichier CSV : resultats_evaluation.csv ✅
```

## Configuration des Logs

Le script est configuré pour afficher uniquement les messages d'information et masquer les messages non pertinents provenant de bibliothèques externes, comme MediaPipe et TensorFlow.

## Export des Résultats

Les résultats sont exportés dans un fichier CSV appelé `resultats_evaluation.csv` et incluent les colonnes suivantes :

- **nom** : Nom de l'étudiant
- **prénom** : Prénom de l'étudiant
- **groupe** : Groupe de l'étudiant
- **introduction_score** : Score pour la qualité de l'introduction
- **star_authenticity_score** : Score pour l'authenticité et la méthode STAR
- **communication_score** : Score pour la prise de parole et la communication
- **competence** : Compétence choisie par l'étudiant
- **question** : Question posée par l'étudiant

## Avertissements et Limitations

- **Utilisation de l'API OpenAI** : Assurez-vous que vous avez une clé API valide et des crédits suffisants, car chaque appel à l'API est payant.
- **Compatibilité MediaPipe** : MediaPipe et TensorFlow peuvent générer des avertissements sur certaines configurations de GPU/CPU. Ce script masque ces avertissements, mais ils peuvent parfois apparaître.

## Auteur

Mathieu - Contribeur principal de ce projet de traitement et d'analyse vidéo automatisé.
