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
import spacy
import openai
import warnings

from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv

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

    video_formats = ('.mp4', '.avi', '.mov', '.mkv', '.MOV')
    for video_file in os.listdir(input_folder):
        if video_file.endswith(video_formats):
            input_video_path = os.path.join(input_folder, video_file)
            output_video_path = os.path.join(output_folder, video_file)
            logging.info(f"Traitement de la vidéo : {video_file}")
            compress_video(input_video_path, output_video_path)

def extract_names_from_videos(video_folder):
    """
    Extrait les noms, prénoms et groupes à partir des noms de fichiers vidéo.

    Args:
        video_folder (str): Dossier contenant les vidéos.

    Returns:
        pd.DataFrame: DataFrame contenant les noms, prénoms, groupes et noms de fichiers vidéo.
    """
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov', 'MOV'))]
    data = []
    for video in video_files:
        base_name = os.path.splitext(video)[0]  # Enlever l'extension
        if '_' in base_name:
            parts = base_name.split('_')
            if len(parts) == 3:
                nom, prenom, groupe = parts  # Extraire Nom, Prénom et Groupe
            else:
                nom, prenom, groupe = parts[0], parts[1], ''  # Si le groupe est manquant
        else:
            nom, prenom, groupe = base_name, '', ''  # Si pas de séparateur '_'

        data.append({'video_file': video, 'nom': nom, 'prénom': prenom, 'groupe': groupe})

    # Créer un DataFrame
    df_videos = pd.DataFrame(data)
    logging.info("Extraction des noms, prénoms et groupes terminée.")
    return df_videos

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

def extract_audio(video_path, output_audio_path):
    """
    Extrait l'audio d'une vidéo.

    Args:
        video_path (str): Chemin de la vidéo.
        output_audio_path (str): Chemin du fichier audio de sortie.
    """
    if not os.path.exists(output_audio_path):
        command = f"ffmpeg -i {video_path} -q:a 0 -map a {output_audio_path} -y"
        subprocess.call(command, shell=True)
        logging.info(f"Audio extrait : {output_audio_path}")
    else:
        logging.info(f"Audio déjà existant : {output_audio_path}")

def analyze_pitch_variation(audio_path):
    """
    Analyse les variations de pitch dans un fichier audio.

    Args:
        audio_path (str): Chemin du fichier audio.

    Returns:
        tuple: Moyenne et écart-type du pitch.
    """
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

# def evaluate_transcript_with_openai(transcript):
#     """
#     Évalue un transcript en utilisant l'API OpenAI.

#     Args:
#         transcript (str): Texte transcrit.

#     Returns:
#         dict: Scores obtenus de l'évaluation.
#     """

#     messages = [
#         {"role": "system", "content": "You are a helpful assistant. Your task is to evaluate the following student transcript."},
#         {
#             "role": "user",
#             "content": f"""
#             Voici une transcription de présentation d'un étudiant :

#             "{transcript}"

#             1. Évaluez la qualité de l'introduction en fonction des critères suivants :
#             - Présentation de l'entreprise ciblée et du poste visé,
#             - Choix d'une compétence psychosociale (soft skill) requise par le poste, et justification de l'importance de cette compétence,
#             - Formulation d'une question pertinente en lien avec cette compétence, en se mettant à la place du recruteur.
#             Donnez un score sur 3 :
#             - 1 pour la présentation de l'entreprise et du poste,
#             - 1 pour le choix et la justification de la compétence,
#             - 1 pour la question liée à la compétence.

#             2. Vérifiez si la méthode STAR (Situation, Tâche, Action, Résultat) est respectée dans la réponse. Donnez un score sur 4 :
#             - 1 pour la Situation (description du contexte),
#             - 1 pour la Tâche (explication des responsabilités et objectifs),
#             - 1 pour l'Action (détails des actions entreprises),
#             - 1 pour le Résultat (résultats obtenus et démonstration de la compétence).

#             3. Évaluez l'authenticité et la personnalisation de la réponse. Est-ce que l'étudiant donne des détails spécifiques montrant une réflexion personnelle ? Donnez un score sur 1 si la réponse semble authentique, 0 sinon.

#             Fournissez uniquement les scores dans ce format :
#             - introduction_score: X
#             - intro_company_score: Y
#             - intro_skill_score: Z
#             - intro_question_score: W
#             - star_situation_score: S
#             - star_task_score: T
#             - star_action_score: U
#             - star_result_score: V
#             - authenticity_score: A
#             """
#         }
#     ]


#     completion = openai.ChatCompletion.create(
#     model="gpt-4o-mini",
#     messages=messages,
#     max_tokens=300,
#     temperature=0
# )



#     response_text = completion.choices[0].message.content.strip()
#     scores = {
#         'introduction_score': 0,
#         'intro_company_score': 0,
#         'intro_skill_score': 0,
#         'intro_question_score': 0,
#         'star_situation_score': 0,
#         'star_task_score': 0,
#         'star_action_score': 0,
#         'star_result_score': 0,
#         'authenticity_score': 0
#     }

#     for line in response_text.split('\n'):
#         for key in scores.keys():
#             if key in line:
#                 scores[key] = float(line.split(":")[1].strip())

#     logging.info("Évaluation du transcript terminée avec OpenAI.")
#     return scores

def evaluate_transcript_with_openai(transcript):
    """
    Évalue un transcript en utilisant l'API OpenAI et extrait la compétence et la question.

    Args:
        transcript (str): Texte transcrit.

    Returns:
        dict: Scores obtenus de l'évaluation, ainsi que la compétence et la question identifiées.
    """

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Your task is to evaluate the following student transcript according to specific criteria."},
        {
            "role": "user",
            "content": f"""
            Voici une transcription de présentation d'un étudiant :

            "{transcript}"

            Évaluez cette transcription selon les critères suivants :

            1. Introduction (1 point) :
               - La justification du choix de la compétence est-elle claire et convaincante ?
               - La question inventée est-elle en lien direct avec la compétence choisie ?
               Donnez un score sur 1 si les deux éléments sont respectés, sinon 0.

            2. Méthode STAR et Authenticité (1 point) :
               - La réponse suit-elle les 4 étapes de la méthode STAR (Situation, Tâche, Action, Résultat) de manière fluide, sans confusion ou omissions majeures ?
               - La réponse est-elle originale, personnalisée et authentique, avec des détails spécifiques et réalistes montrant que l’étudiant a réfléchi à son expérience, et non une réponse générique ?
               Donnez un score sur 1 si les deux éléments sont respectés, sinon 0.

            3. Prise de parole et Communication (1 point) :
               - Le langage corporel est-il approprié (gestes, contact visuel, posture et mouvements) pour soutenir et renforcer le discours sans être distrayant ?
               - Le ton et le rythme de la voix sont-ils adaptés à la situation, régulier, ni trop rapide ni trop lent, permettant une bonne compréhension ?
               Donnez un score sur 1 si les deux éléments sont respectés, sinon 0.

            De plus, identifiez la compétence mentionnée et la question posée par l'étudiant dans l'introduction, et fournissez-les dans ce format :

            - competence: [nom de la compétence]
            - question: [question posée]
            Fournissez uniquement les scores et les informations dans ce format :
            - introduction_score: X
            - star_authenticity_score: Y
            - communication_score: Z
            - competence: [nom de la compétence]
            - question: [question posée]
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
    scores = {
        'introduction_score': 0,
        'star_authenticity_score': 0,
        'communication_score': 0,
        'competence': "",
        'question': ""
    }

    # Extraire les scores et les informations de la réponse
    for line in response_text.split('\n'):
        for key in scores.keys():
            if key in line:
                if key in ['competence', 'question']:
                    scores[key] = line.split(":")[1].strip()
                else:
                    scores[key] = float(line.split(":")[1].strip())

    logging.info("Évaluation du transcript terminée avec OpenAI ✅.")
    return scores


def main():
    # Chemins des dossiers
    video_folder = "data/videos/"
    output_video_folder = "data/videos_compressed/"
    audio_folder = "data/audios/"

    # Compression des vidéos
    logging.info("Début de la compression des vidéos.")
    compress_all_videos(video_folder, output_video_folder)
    logging.info("Compression des vidéos terminée ✅.")

    # Extraction des noms
    df_videos = extract_names_from_videos(output_video_folder)

    # Transcription avec Whisper
    logging.info("Début de la transcription avec Whisper.")
    whisper_model_name = "base"
    whisper_model = whisper.load_model(whisper_model_name)
    df_videos['transcript'] = df_videos['video_file'].apply(
        lambda video_file: transcribe_video(whisper_model, os.path.join(output_video_folder, video_file))
    )
    logging.info("Transcription terminée ✅.")

    # Analyse non-verbale avec MediaPipe
    logging.info("Début de l'analyse non-verbale avec MediaPipe.")
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
        lambda video_file: analyze_video_key_landmarks(
            os.path.join(output_video_folder, video_file), key_landmarks, pose)
    )
    df_videos['movements'], df_videos['visibilities'] = zip(*df_videos['pose_data'].apply(
        lambda pose_data: calculate_movement_and_visibility(pose_data, key_landmarks)
    ))
    logging.info("Analyse non-verbale terminée ✅ ")

    # Mesure du débit de parole
    logging.info("Calcul du débit de parole.")
    df_videos['speech_rate'] = df_videos.apply(
        lambda row: calculate_speech_rate(
            row['transcript'],
            get_video_duration(os.path.join(video_folder, row['video_file']))
        ),
        axis=1
    )
    logging.info("Calcul du débit de parole terminé ✅.")

    # Mesure du ton
    logging.info("Début de l'analyse du ton.")
    if not os.path.exists(audio_folder):
        os.makedirs(audio_folder)
        logging.info(f"Dossier créé : {audio_folder}")

    df_videos['audio_file'] = df_videos['video_file'].apply(
        lambda video_file: video_file.replace('.mp4', '.wav')
    )

    for video_file, audio_file in zip(df_videos['video_file'], df_videos['audio_file']):
        extract_audio(os.path.join(output_video_folder, video_file), os.path.join(audio_folder, audio_file))

    df_videos['pitch_mean'], df_videos['pitch_std'] = zip(*df_videos['audio_file'].apply(
        lambda audio_file: analyze_pitch_variation(os.path.join(audio_folder, audio_file))
    ))
    logging.info("Analyse du ton terminée ✅.")

    # Calcul des scores globaux
    logging.info("Calcul des scores globaux.")
    scaler = MinMaxScaler()
    df_score = df_videos[['nom', 'prénom']].copy()

    # Score de mouvement
    df_score['movement_score'] = df_videos.apply(
        lambda row: calculate_movement_rate(row['movements'], row['visibilities'], key_landmarks), axis=1
    )
    df_score['movement_score'] = scaler.fit_transform(df_score[['movement_score']])

    # Score de débit de parole
    df_score['speech_score'] = scaler.fit_transform(df_videos[['speech_rate']])

    # Score de ton
    pitch_mean_normalized = scaler.fit_transform(df_videos[['pitch_mean']])
    pitch_std_normalized = scaler.fit_transform(df_videos[['pitch_std']])
    df_score['tone_score'] = 0.5 * pitch_mean_normalized + 0.5 * pitch_std_normalized

    # Évaluation du transcript avec GPT
    df_scores_expanded = df_videos['transcript'].apply(evaluate_transcript_with_openai)
    df_scores_expanded = pd.json_normalize(df_scores_expanded)
    df_score = pd.concat([df_score, df_scores_expanded], axis=1)
    logging.info("Calcul des scores globaux terminé ✅.")

    # Affichage du DataFrame final
    print(df_score.head())

    # Exporter le DataFrame final en fichier CSV
    output_path = "resultats_evaluation.csv"
    df_score.to_csv(output_path, index=False, encoding='utf-8')
    logging.info(f"DataFrame exporté en fichier CSV : {output_path} ✅.")

if __name__ == "__main__":
    main()
