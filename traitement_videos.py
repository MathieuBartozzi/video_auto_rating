import absl.logging
import os
import subprocess
import logging
import whisper
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import librosa
import openai
import warnings
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import re


# Configuration du logging principal pour votre script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ignorer tous les avertissements Python
warnings.filterwarnings("ignore")

# Réduire la verbosité des logs de MediaPipe (utilisant absl) pour n'afficher que les erreurs
absl.logging.set_verbosity(absl.logging.ERROR)

# Masquer les logs TensorFlow de niveau inférieur pour n'afficher que les erreurs critiques
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = tous les logs, 1 = infos, 2 = avertissements, 3 = erreurs uniquement
# Exemple d'autres suppressions de logs
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # Masque les logs d'absl utilisés par Mediapipe

# Charger les variables d'environnement
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compress_video(input_path, output_path):
    """
    Compresse une vidéo en utilisant FFmpeg.

    Args:
        input_path (str): Chemin du fichier vidéo d'entrée.
        output_path (str): Chemin du fichier vidéo de sortie.
    """
    if not os.path.exists(output_path):
        command = [
            "ffmpeg", "-i", input_path,
            "-vcodec", "libx264",
            "-crf", "28",
            "-preset", "fast",
            "-acodec", "aac", "-strict", "-2",
            output_path
        ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info(f"Vidéo compressée : {output_path}")
    else:
        logging.info(f"Vidéo déjà compressée : {output_path}")

def compress_all_videos(input_folder, output_folder):
    """
    Compresse toutes les vidéos dans un dossier donné.

    Args:
        input_folder (str): Dossier contenant les vidéos à compresser.
        output_folder (str): Dossier où les vidéos compressées seront enregistrées.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logging.info(f"Dossier créé : {output_folder}")

    video_formats = ('.mp4', '.avi', '.mov', '.mkv', '.MOV','.MP4')
    for video_file in os.listdir(input_folder):
        if video_file.endswith(video_formats):
            input_video_path = os.path.join(input_folder, video_file)
            output_video_path = os.path.join(output_folder, video_file)
            logging.info(f"Traitement de la vidéo : {video_file}")
            compress_video(input_video_path, output_video_path)
            os.remove(input_video_path)
            logging.info(f"Vidéo originale supprimée : {input_video_path}")

# def extract_names_from_videos(video_folder):
#     """
#     Extrait les noms, prénoms et groupes à partir des noms de fichiers vidéo.

#     Args:
#         video_folder (str): Dossier contenant les vidéos.

#     Returns:
#         pd.DataFrame: DataFrame contenant les noms, prénoms, groupes et noms de fichiers vidéo.
#     """
#     video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov', 'MOV'))]
#     data = []
#     for video in video_files:
#         base_name = os.path.splitext(video)[0]  # Enlever l'extension
#         if '_' in base_name:
#             parts = base_name.split('_')
#             if len(parts) == 3:
#                 nom, prenom, groupe = parts  # Extraire Nom, Prénom et Groupe
#             else:
#                 nom, prenom, groupe = parts[0], parts[1], ''  # Si le groupe est manquant
#         else:
#             nom, prenom, groupe = base_name, '', ''  # Si pas de séparateur '_'

#         data.append({'video_file': video, 'nom': nom, 'prénom': prenom, 'groupe': groupe})

#     # Créer un DataFrame
#     df_videos = pd.DataFrame(data)
#     logging.info("Extraction des noms, prénoms et groupes terminée.")
#     return df_videos

def transcribe_video(whisper_model, video_path):
    """
    Transcrit une vidéo en utilisant le modèle Whisper.

    Args:
        whisper_model: Modèle Whisper chargé.
        video_path (str): Chemin de la vidéo à transcrire.

    Returns:
        str: Texte transcrit.
    """
    result = whisper_model.transcribe(video_path)
    logging.info(f"Transcription terminée pour : {video_path}")
    return result['text']

def analyze_video_key_landmarks(video_path, key_landmarks, pose):
    """
    Analyse une vidéo pour extraire les key landmarks avec MediaPipe.

    Args:
        video_path (str): Chemin de la vidéo à analyser.
        key_landmarks (dict): Dictionnaire des key landmarks.
        pose: Instance de MediaPipe Pose.

    Returns:
        list: Liste des données des key landmarks par frame.
    """
    cap = cv2.VideoCapture(video_path)
    pose_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            key_landmarks_data = {
                name: (landmarks[idx].x, landmarks[idx].y, landmarks[idx].visibility)
                for name, idx in key_landmarks.items()
            }
            pose_data.append(key_landmarks_data)
    cap.release()
    logging.info(f"Analyse MediaPipe terminée pour : {video_path}")
    return pose_data

def calculate_movement_and_visibility(pose_data, key_landmarks):
    """
    Calcule les mouvements et la visibilité moyenne des key landmarks.

    Args:
        pose_data (list): Données des key landmarks par frame.
        key_landmarks (dict): Dictionnaire des key landmarks.

    Returns:
        tuple: Dictionnaires des mouvements totaux et des visibilités moyennes.
    """
    total_movement = {landmark: 0 for landmark in key_landmarks}
    visibility_stats = {landmark: [] for landmark in key_landmarks}

    for prev_frame, curr_frame in zip(pose_data[:-1], pose_data[1:]):
        for landmark_name in key_landmarks:
            prev_landmark = prev_frame[landmark_name]
            curr_landmark = curr_frame[landmark_name]
            if prev_landmark[2] > 0.5 and curr_landmark[2] > 0.5:
                dist = np.linalg.norm(np.array(curr_landmark[:2]) - np.array(prev_landmark[:2]))
                total_movement[landmark_name] += dist
            visibility_stats[landmark_name].append(curr_landmark[2])

    avg_visibility = {landmark: np.mean(vis) for landmark, vis in visibility_stats.items()}
    logging.info("Calcul des mouvements et visibilités terminé.")
    return total_movement, avg_visibility

def get_video_duration(video_path):
    """
    Calcule la durée d'une vidéo en minutes.

    Args:
        video_path (str): Chemin de la vidéo.

    Returns:
        float: Durée de la vidéo en minutes.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = frame_count / fps
    cap.release()
    return duration_seconds / 60

def calculate_speech_rate(transcript, video_duration_minutes):
    """
    Calcule le débit de parole (mots par minute).

    Args:
        transcript (str): Texte transcrit.
        video_duration_minutes (float): Durée de la vidéo en minutes.

    Returns:
        float: Débit de parole en mots par minute.
    """
    word_count = len(transcript.split())
    if video_duration_minutes > 0:
        return word_count / video_duration_minutes
    else:
        return 0

# def extract_audio(video_path, output_audio_path):
#     """
#     Extrait l'audio d'une vidéo.

#     Args:
#         video_path (str): Chemin de la vidéo.
#         output_audio_path (str): Chemin du fichier audio de sortie.
#     """
#     if not os.path.exists(output_audio_path):
#         command = f"ffmpeg -i {video_path} -q:a 0 -map a {output_audio_path} -y"
#         subprocess.call(command, shell=True)
#         logging.info(f"Audio extrait : {output_audio_path}")
#     else:
#         logging.info(f"Audio déjà existant : {output_audio_path}")

def extract_audio(video_path, output_audio_path):
    if not os.path.exists(output_audio_path):
        command = f"ffmpeg -i {video_path} -q:a 0 -map a {output_audio_path} -y"
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0 or not os.path.exists(output_audio_path):
            logging.error(f"Erreur FFmpeg pour {video_path}: {result.stderr.decode()}")
        else:
            logging.info(f"Audio extrait : {output_audio_path}")
    else:
        logging.info(f"Audio déjà existant : {output_audio_path}")


# def analyze_pitch_variation(audio_path):
#     """
#     Analyse les variations de pitch dans un fichier audio.

#     Args:
#         audio_path (str): Chemin du fichier audio.

#     Returns:
#         tuple: Moyenne et écart-type du pitch.
#     """
#     y, sr = librosa.load(audio_path)
#     frame_length = 1024
#     hop_length = 256
#     pitches, _, _ = librosa.pyin(
#         y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
#         sr=sr, frame_length=frame_length, hop_length=hop_length
#     )
#     valid_pitches = pitches[~np.isnan(pitches)]
#     if len(valid_pitches) == 0:
#         return 0, 0
#     pitch_mean = np.mean(valid_pitches)
#     pitch_std = np.std(valid_pitches)
#     logging.info(f"Analyse du pitch terminée pour : {audio_path}")
#     return pitch_mean, pitch_std

def analyze_pitch_variation(audio_path):
    """
    Analyse les variations de pitch dans un fichier audio.

    Args:
        audio_path (str): Chemin du fichier audio.

    Returns:
        tuple: Moyenne et écart-type du pitch.
    """
    if not os.path.exists(audio_path):
        logging.error(f"Fichier audio manquant : {audio_path}")
        return 0, 0  # Valeurs par défaut en cas d'erreur

    try:
        y, sr = librosa.load(audio_path)
        frame_length = 1024
        hop_length = 256
        pitches, _, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
            sr=sr, frame_length=frame_length, hop_length=hop_length
        )
        valid_pitches = pitches[~np.isnan(pitches)]
        if len(valid_pitches) == 0:
            return 0, 0
        pitch_mean = np.mean(valid_pitches)
        pitch_std = np.std(valid_pitches)
        logging.info(f"Analyse du pitch terminée pour : {audio_path}")
        return pitch_mean, pitch_std
    except Exception as e:
        logging.error(f"Erreur lors de l'analyse du pitch pour {audio_path}: {e}")
        return 0, 0  # Retourner des valeurs par défaut en cas d'erreur

def calculate_movement_rate(total_movement, avg_visibility, key_landmarks):
    """
    Calcule le score lié au mouvement non verbal.

    Args:
        total_movement (dict): Mouvements totaux des key landmarks.
        avg_visibility (dict): Visibilités moyennes des key landmarks.
        key_landmarks (dict): Dictionnaire des key landmarks.

    Returns:
        float: Score de mouvement.
    """
    stability_weight = 0.3
    gesture_weight = 0.5
    diversity_weight = 0.2

    stability_score = (avg_visibility['left_shoulder'] + avg_visibility['right_shoulder']) / 2
    gesture_score = (total_movement['left_wrist'] + total_movement['right_wrist'] +
                     total_movement['left_elbow'] + total_movement['right_elbow']) / 4

    left_side_movement = total_movement['left_wrist'] + total_movement['left_elbow']
    right_side_movement = total_movement['right_wrist'] + total_movement['right_elbow']

    if max(left_side_movement, right_side_movement) > 0:
        diversity_score = min(left_side_movement, right_side_movement) / max(left_side_movement, right_side_movement)
    else:
        diversity_score = 0

    movement_rate = (stability_score * stability_weight +
                     gesture_score * gesture_weight +
                     diversity_score * diversity_weight)
    logging.info("Calcul du score de mouvement terminé.")
    return movement_rate


def evaluate_transcript_with_openai(transcript):
    """
    Évalue un transcript en utilisant l'API OpenAI et extrait la compétence et la question.

    Args:
        transcript (str): Texte transcrit.

    Returns:
        dict: Scores obtenus de l'évaluation, ainsi que la compétence et la question identifiées.
    """

    messages = [
            {"role": "system", "content": "You are a helpful assistant. Your task is to evaluate the following student transcript."},
            {
                "role": "user",
                "content": f"""
                Voici une transcription de présentation d'un étudiant :

                "{transcript}"

                1. Évaluez la qualité de l'introduction en fonction des critères suivants :
            - Présentation de l'entreprise ciblée et du poste visé,
            - Choix d'une compétence psychosociale (soft skill) requise par le poste, et justification de l'importance de cette compétence,
            - Formulation d'une question pertinente en lien avec cette compétence, en se mettant à la place du recruteur.

            Donnez les 8 scores suivants (0 ou 1) :

            1. intro_company_score : Présentation claire de l'entreprise et du poste.
            2. intro_skill_score : Justification du choix de la compétence.
            3. intro_question_score : Pertinence de la question liée à la compétence.

            Vérifiez si la méthode STAR (Situation, Tâche, Action, Résultat) est respectée dans la réponse :
            4. star_situation_score : Description correcte du contexte.
            5. star_task_score : Explications des responsabilités et objectifs.
            6. star_action_score : Détails des actions entreprises.
            7. star_result_score : Résultats démontrant une maîtrise de la compétence.

            Évaluez l'authenticité et la personnalisation de la réponse :
            8. authenticity_score : Réponse authentique et réflexion personnelle.

            Fournissez les scores dans ce format (sans autres commentaires) :
            - intro_company_score: X
            - intro_skill_score: Y
            - intro_question_score: Z
            - star_situation_score: W
            - star_task_score: S
            - star_action_score: T
            - star_result_score: U
            - authenticity_score: V
            """
            }
        ]

    # Appel de l'API OpenAI pour générer la complétion
    completion = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=300,
        temperature=0
    )

    response_text = completion.choices[0].message.content.strip()

    # Initialiser le dictionnaire des scores
    # Initialisation des scores
    scores = {
        'intro_company_score': 0,
        'intro_skill_score': 0,
        'intro_question_score': 0,
        'star_situation_score': 0,
        'star_task_score': 0,
        'star_action_score': 0,
        'star_result_score': 0,
        'authenticity_score': 0
    }


    # Extraire les scores et les informations de la réponse
    for line in response_text.split('\n'):
        for key in scores.keys():
            if key in line:
                if key in ['competence', 'question']:
                    scores[key] = line.split(":")[1].strip()
                else:
                    scores[key] = float(line.split(":")[1].strip())


    # for line in response_text.split('\n'):
    #     line = line.strip()
    #     for key in scores.keys():
    #         if line.startswith(key):  # Vérifie si la ligne commence par la clé
    #             try:
    #                 # Extraction sécurisée de la valeur après ":"
    #                 value = line.split(":", 1)[1].strip()
    #                 scores[key] = int(value)  # Convertir en entier (0 ou 1 attendu)
    #             except (IndexError, ValueError):
    #                 logging.warning(f"Impossible de traiter la ligne : {line}")
    logging.info("Évaluation du transcript terminée avec OpenAI ✅.")
    return scores


def clean_filename(filename):
    """
    Nettoie le nom de fichier en remplaçant les espaces et caractères spéciaux.
    """
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)

def rename_videos_and_update_csv(input_folder, csv_path):
    """
    Renomme les vidéos dans un dossier et met à jour le CSV avec les nouveaux noms.
    """
    # Charger le CSV
    df_videos = pd.read_csv(csv_path, sep=";").copy()

    updated_rows = 0
    for i, row in df_videos.iterrows():
        original_name = row['video_file']
        original_path = os.path.join(input_folder, original_name)

        # Vérifier si le fichier existe
        if os.path.exists(original_path):
            new_name = clean_filename(original_name)
            new_path = os.path.join(input_folder, new_name)

            # Renommer le fichier uniquement si nécessaire
            if original_name != new_name:
                os.rename(original_path, new_path)
                logging.info(f"Renommé : {original_name} -> {new_name}")
                df_videos.at[i, 'video_file'] = new_name
                updated_rows += 1

    # Sauvegarder le CSV mis à jour
    if updated_rows > 0:
        df_videos.to_csv(csv_path, sep=";", index=False)
        logging.info(f"{updated_rows} noms de fichiers mis à jour dans le CSV.")
    else:
        logging.info("Aucun nom de fichier n'a été modifié.")

def save_progress(df, path="data/df_videos_progress.csv"):
    """
    Sauvegarde progressive du DataFrame dans un fichier CSV.

    Args:
        df (pd.DataFrame): DataFrame à sauvegarder.
        path (str): Chemin du fichier CSV de sauvegarde.
    """
    df.to_csv(path, index=False, sep=";", encoding='utf-8')
    logging.info(f"Progression sauvegardée dans {path}")



# def main():
#     # Chemins des dossiers
#     video_folder = "data/videos/"
#     # video_folder = "data/videos_compressed/"
#     audio_folder = "data/audios/"
#     csv_path = "data/students.csv"
#     progress_path = "data/df_videos_progress.csv"

#         # Rechargement de la progression si elle existe
#     if os.path.exists(progress_path):
#         df_videos = pd.read_csv(progress_path, sep=";")
#         logging.info("Progression existante rechargée.")
#     else:
#         df_videos = pd.read_csv(csv_path, sep=";")
#         logging.info("Aucune sauvegarde existante. Chargement initial du CSV.")

#     # Étape 1 : Extraction audio
#     if 'audio_file' not in df_videos.columns:
#         df_videos['audio_file'] = df_videos['video_file'].apply(
#             lambda video_file: os.path.splitext(video_file)[0] + '.wav'
#         )
#         save_progress(df_videos, progress_path)  # Sauvegarde après étape
#         logging.info("Étape d'extraction audio initialisée.")

#     for video_file, audio_file in zip(df_videos['video_file'], df_videos['audio_file']):
#         audio_path = os.path.join(audio_folder, audio_file)
#         if not os.path.exists(audio_path):
#             extract_audio(os.path.join(video_folder, video_file), audio_path)
#         else:
#             logging.info(f"Audio déjà existant : {audio_path}")

#     save_progress(df_videos, progress_path)  # Sauvegarde après extraction audio

#     # Étape 2 : Transcription avec Whisper
#     if 'transcript' not in df_videos.columns:
#         whisper_model = whisper.load_model("base")
#         df_videos['transcript'] = df_videos['video_file'].apply(
#             lambda video_file: transcribe_video(whisper_model, os.path.join(video_folder, video_file))
#         )
#         save_progress(df_videos, progress_path)
#         logging.info("Étape de transcription terminée ✅.")

#     # Étape 3 : Analyse des pitchs
#     if 'pitch_mean' not in df_videos.columns or 'pitch_std' not in df_videos.columns:
#         df_videos['pitch_mean'], df_videos['pitch_std'] = zip(*df_videos['audio_file'].apply(
#             lambda audio_file: analyze_pitch_variation(os.path.join(audio_folder, audio_file))
#         ))
#         save_progress(df_videos, progress_path)
#         logging.info("Étape d'analyse du pitch terminée ✅.")


#     rename_videos_and_update_csv(video_folder, csv_path)


#     # Compression des vidéos
#     # logging.info("Début de la compression des vidéos.")
#     # # compress_all_videos(video_folder, video_folder)
#     # logging.info("Compression des vidéos terminée ✅.")

#     # Extraction des noms
#     # df_videos = extract_names_from_videos(output_video_folder)
#     # Charger le fichier CSV
#     if not os.path.exists(csv_path):
#         logging.error(f"Le fichier {csv_path} est introuvable.")
#         return

#     # df_videos = pd.read_csv(csv_path,sep=";").copy()

#     # Vérifiez si la colonne "video_file" existe dans le CSV
#     if 'video_file' not in df_videos.columns:
#         logging.error("La colonne 'video_file' est introuvable dans le fichier CSV.")
#         return


#     # Obtenir toutes les vidéos présentes dans le dossier
#     video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.MP4', '.avi', '.mov', '.mkv', '.MOV'))]

#     # Vidéos attendues depuis le CSV
#     videos_in_csv = df_videos['video_file'].tolist()

#     # Identifier les vidéos manquantes
#     missing_videos = set(videos_in_csv) - set(video_files)

#     if missing_videos:
#         logging.error(f"Les vidéos suivantes, référencées dans le fichier CSV, sont manquantes dans le dossier : {missing_videos}")
#         raise FileNotFoundError("Certaines vidéos du CSV ne sont pas présentes dans le dossier. Arrêt du traitement.")

#     # Filtrer les vidéos du CSV qui existent dans le dossier
#     df_videos = df_videos[df_videos['video_file'].isin(video_files)]
#     logging.info(f"{len(df_videos)} vidéos trouvées pour traitement.")


#     # Transcription avec Whisper
#     logging.info("Début de la transcription avec Whisper.")
#     whisper_model_name = "base"
#     whisper_model = whisper.load_model(whisper_model_name)
#     df_videos['transcript'] = df_videos['video_file'].apply(
#         lambda video_file: transcribe_video(whisper_model, os.path.join(video_folder, video_file))
#     )
#     logging.info("Transcription terminée ✅.")

#     # Analyse non-verbale avec MediaPipe
#     logging.info("Début de l'analyse non-verbale avec MediaPipe.")
#     mp_pose = mp.solutions.pose
#     pose = mp_pose.Pose()
#     key_landmarks = {
#         'left_wrist': 15,
#         'right_wrist': 16,
#         'left_elbow': 13,
#         'right_elbow': 14,
#         'left_shoulder': 11,
#         'right_shoulder': 12
#     }
#     df_videos['pose_data'] = df_videos['video_file'].apply(
#         lambda video_file: analyze_video_key_landmarks(
#             os.path.join(video_folder, video_file), key_landmarks, pose)
#     )
#     df_videos['movements'], df_videos['visibilities'] = zip(*df_videos['pose_data'].apply(
#         lambda pose_data: calculate_movement_and_visibility(pose_data, key_landmarks)
#     ))
#     logging.info("Analyse non-verbale terminée ✅ ")

#     # Mesure du débit de parole
#     logging.info("Calcul du débit de parole.")
#     df_videos['speech_rate'] = df_videos.apply(
#         lambda row: calculate_speech_rate(
#             row['transcript'],
#             get_video_duration(os.path.join(video_folder, row['video_file']))
#         ),
#         axis=1
#     )
#     logging.info("Calcul du débit de parole terminé ✅.")

#     # Mesure du ton
#     logging.info("Début de l'analyse du ton.")
#     if not os.path.exists(audio_folder):
#         os.makedirs(audio_folder)
#         logging.info(f"Dossier créé : {audio_folder}")

#     df_videos['audio_file'] = df_videos['video_file'].apply(
#     lambda video_file: os.path.splitext(video_file)[0] + '.wav'
#     )

#     for video_file, audio_file in zip(df_videos['video_file'], df_videos['audio_file']):
#         extract_audio(os.path.join(video_folder, video_file), os.path.join(audio_folder, audio_file))

#     df_videos['pitch_mean'], df_videos['pitch_std'] = zip(*df_videos['audio_file'].apply(
#         lambda audio_file: analyze_pitch_variation(os.path.join(audio_folder, audio_file))
#     ))
#     logging.info("Analyse du ton terminée ✅.")

#     # Calcul des scores globaux
#     logging.info("Calcul des scores globaux.")
#     scaler = MinMaxScaler()

#     # Score de mouvement
#     df_videos['movement_score'] = df_videos.apply(
#         lambda row: calculate_movement_rate(row['movements'], row['visibilities'], key_landmarks), axis=1
#     )
#     df_videos['movement_score'] = scaler.fit_transform(df_videos[['movement_score']])

#     # Score de débit de parole
#     df_videos['speech_score'] = scaler.fit_transform(df_videos[['speech_rate']])

#     # Score de ton
#     pitch_mean_normalized = scaler.fit_transform(df_videos[['pitch_mean']])
#     pitch_std_normalized = scaler.fit_transform(df_videos[['pitch_std']])
#     df_videos['tone_score'] = 0.5 * pitch_mean_normalized + 0.5 * pitch_std_normalized

#     # Appliquer la fonction pour évaluer les transcripts
#     df_videos = df_videos.join(
#         df_videos['transcript'].apply(evaluate_transcript_with_openai).apply(pd.Series)
#     )
#     logging.info("Calcul des scores globaux terminé ✅.")


#     # Affichage du DataFrame final
#     print(df_videos.head())
#     # Liste des colonnes à supprimer
#     colonnes_a_supprimer = [
#         "Pass/Fail", "Commentaire", "pose_data", "movements",'speech_rate',
#         "visibilities", "audio_file", "pitch_mean", "pitch_std"
#     ]

#     # Suppression des colonnes
#     df_videos_clean = df_videos.drop(columns=colonnes_a_supprimer, errors='ignore')

#     # Exporter le DataFrame final en fichier CSV
#     output_path = "resultats_evaluation.csv"
#     df_videos_clean.to_csv(output_path, index=False, encoding='utf-8')

#     logging.info(f"DataFrame exporté en fichier CSV : {output_path} ✅.")

# if __name__ == "__main__":
#     main()

def main():
    # Chemins des dossiers
    video_folder = "data/videos/"
    audio_folder = "data/audios/"
    csv_path = "data/students.csv"
    progress_path = "data/df_videos_progress.csv"

    # Rechargement de la progression si elle existe
    if os.path.exists(progress_path):
        df_videos = pd.read_csv(progress_path, sep=";")
        logging.info("Progression existante rechargée.")
    else:
        df_videos = pd.read_csv(csv_path, sep=";")
        logging.info("Aucune sauvegarde existante. Chargement initial du CSV.")

        # Ajout de la colonne audio_file au départ
        df_videos['audio_file'] = df_videos['video_file'].apply(
            lambda video_file: os.path.splitext(video_file)[0] + '.wav'
        )
        save_progress(df_videos, progress_path)
     # Étape 0 : Gestion des vidéos manquantes
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.MP4', '.avi', '.mov', '.mkv', '.MOV'))]
    videos_in_csv = df_videos['video_file'].tolist()

    # Identifier les vidéos manquantes
    missing_videos = set(videos_in_csv) - set(video_files)

    if missing_videos:
        logging.warning(f"Les vidéos suivantes, référencées dans le fichier CSV, sont manquantes : {missing_videos}")
        # Supprimer les lignes correspondantes dans le DataFrame
        df_videos = df_videos[~df_videos['video_file'].isin(missing_videos)]
        save_progress(df_videos, progress_path)  # Sauvegarde après nettoyage
        logging.info(f"{len(missing_videos)} vidéos manquantes supprimées du traitement.")

    # Filtrer les vidéos existantes
    df_videos = df_videos[df_videos['video_file'].isin(video_files)]
    logging.info(f"{len(df_videos)} vidéos restantes pour traitement.")

    # Étape 1 : Extraction audio
    for video_file, audio_file in zip(df_videos['video_file'], df_videos['audio_file']):
        audio_path = os.path.join(audio_folder, audio_file)
        if not os.path.exists(audio_path):
            extract_audio(os.path.join(video_folder, video_file), audio_path)
    save_progress(df_videos, progress_path)

    # Étape 2 : Transcription avec Whisper
    if 'transcript' not in df_videos.columns:
        whisper_model = whisper.load_model("base")
        df_videos['transcript'] = df_videos['video_file'].apply(
            lambda video_file: transcribe_video(whisper_model, os.path.join(video_folder, video_file))
        )
        save_progress(df_videos, progress_path)

    # Étape 3 : Analyse des pitchs
    if 'pitch_mean' not in df_videos.columns or 'pitch_std' not in df_videos.columns:
        df_videos['pitch_mean'], df_videos['pitch_std'] = zip(*df_videos['audio_file'].apply(
            lambda audio_file: analyze_pitch_variation(os.path.join(audio_folder, audio_file))
        ))
        save_progress(df_videos, progress_path)

    # Étape 4 : Analyse non-verbale avec MediaPipe
    if 'pose_data' not in df_videos.columns:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        key_landmarks = {
            'left_wrist': 15,
            'right_wrist': 16,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_shoulder': 11,
            'right_shoulder': 12
        }
        df_videos['pose_data'] = df_videos['video_file'].apply(
            lambda video_file: analyze_video_key_landmarks(os.path.join(video_folder, video_file), key_landmarks, pose)
        )
        df_videos['movements'], df_videos['visibilities'] = zip(*df_videos['pose_data'].apply(
            lambda pose_data: calculate_movement_and_visibility(pose_data, key_landmarks)
        ))
        save_progress(df_videos, progress_path)

    # Étape 5 : Calcul des scores
    if 'speech_rate' not in df_videos.columns:
        df_videos['speech_rate'] = df_videos.apply(
            lambda row: calculate_speech_rate(
                row['transcript'], get_video_duration(os.path.join(video_folder, row['video_file']))
            ),
            axis=1
        )
        save_progress(df_videos, progress_path)

    # Normalisation et calcul final des scores
    scaler = MinMaxScaler()
    df_videos['movement_score'] = df_videos.apply(
        lambda row: calculate_movement_rate(row['movements'], row['visibilities'], key_landmarks), axis=1
    )
    df_videos['movement_score'] = scaler.fit_transform(df_videos[['movement_score']])
    df_videos['speech_score'] = scaler.fit_transform(df_videos[['speech_rate']])

    pitch_mean_norm = scaler.fit_transform(df_videos[['pitch_mean']])
    pitch_std_norm = scaler.fit_transform(df_videos[['pitch_std']])
    df_videos['tone_score'] = 0.5 * pitch_mean_norm + 0.5 * pitch_std_norm

    # Évaluation des transcripts avec OpenAI
    if 'intro_company_score' not in df_videos.columns:  # Vérification si déjà fait
        df_videos = df_videos.join(
            df_videos['transcript'].apply(evaluate_transcript_with_openai).apply(pd.Series)
        )
        save_progress(df_videos, progress_path)

    # Nettoyage des colonnes inutiles
    colonnes_a_supprimer = [
        "Pass/Fail", "Commentaire", "pose_data", "movements", "speech_rate",
        "visibilities", "audio_file", "pitch_mean", "pitch_std"
    ]
    df_videos_clean = df_videos.drop(columns=colonnes_a_supprimer, errors='ignore')

    # Export final
    output_path = "resultats_evaluation.csv"
    df_videos_clean.to_csv(output_path, index=False, encoding='utf-8')
    logging.info(f"DataFrame exporté en fichier CSV : {output_path} ✅.")

if __name__ == "__main__":
    main()
