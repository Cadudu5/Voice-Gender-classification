import librosa
import pandas as pd
import numpy as np
import os

desired_length = 5

def extract_important_features(audio_file, classification):


    nome = str(audio_file)
    if classification == 'male':
        nome = nome[48:]
    else:
        nome = nome[50:]
    """
    Extracts important features from an audio file using librosa.
    Args:
        audio_file (str): Path to the audio file (e.g., MP3, WAV).
    Returns:
        dict: A dictionary containing the extracted features.
            - 'mel_spectrogram': Mel spectrogram
            - 'mfccs': Mel-Frequency Cepstral Coefficients
            - 'spectral_centroid': Spectral centroid
            - 'spectral_bandwidth': Spectral bandwidth
            - 'zero_crossing_rate': Zero-crossing rate
    """
    y, sr = librosa.load(audio_file)

      # Padroniza o comprimento para 20 segundos
    target_length = int(desired_length * sr)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y))) # completa com 0 se for menor que 5s
    elif len(y) > target_length:
        y = y[:target_length]

    # Extract features
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)



    # Create a dictionary to store the features
    features = {
        
        'audio-name': nome,
        'class': classification,
        'mel_spectrogram': np.mean(mel_spectrogram),
        'spectral_centroid': np.mean(spectral_centroid),
        'spectral_bandwidth': np.mean(spectral_bandwidth),
        'zero_crossing_rate': np.mean(zero_crossing_rate),
    }

    for i, mfcc in enumerate(mfccs):
        features[f'mfcc_{i + 1}'] = np.mean(mfcc)

    return features


# Define folder paths
male_folder = "C:\\Users\\cadud\\Documents\\RNA\\voice samples\\male"
female_folder = "C:\\Users\\cadud\\Documents\\RNA\\voice samples\\female"

output = 'masculino.csv'
output2 = 'feminino.csv'

male_data = []
# Iterate through each file in the folder
for filename in os.listdir(male_folder):
    file_path = os.path.join(male_folder, filename)

    # Check if the item is a file (not a subdirectory)
    if os.path.isfile(file_path):
        # Extract MFCCs
        # Cria um DataFrame do Pandas com os dados
        male_data.append(extract_important_features(f'{file_path}', 'male'))
df = pd.DataFrame(male_data)
df.info()
# Salva o DataFrame em um arquivo CSV
df.to_csv(output, index=False)


female_data = []
# Iterate through each file in the folder
for filename in os.listdir(female_folder):
    file_path = os.path.join(female_folder, filename)

    # Check if the item is a file (not a subdirectory)
    if os.path.isfile(file_path):
        # Extract MFCCs
        # Cria um DataFrame do Pandas com os dados
       female_data.append(extract_important_features(f'{file_path}', 'female'))
df = pd.DataFrame(female_data)
df.info()
# Salva o DataFrame em um arquivo CSV
df.to_csv(output2, index=False)

