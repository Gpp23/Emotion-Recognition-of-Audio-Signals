from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import librosa
import os
import csv
import warnings
import scipy
warnings.filterwarnings("ignore")

class Extractor(object):
    def __init__(self, emotions = "arrabbiato calmo cupo drammatico felice motivante romantico triste vivace", mfccs_n = 40):
        Extractor.__emotions = emotions.split()
        Extractor.__mfccs_n = mfccs_n
        Extractor.__header = Extractor.__headerMaker(self)
        
    def __headerMaker(self):
        header = ""

        for i in range(0, 12):
            header += "chroma_mean_" + str(i) + " "
        for i in range(0, 12):
            header += "chroma_std_" + str(i) + " "

        for i in range(0, Extractor.__mfccs_n):
            header += "mfccs_mean_" + str(i) + " "
        for i in range(0, Extractor.__mfccs_n):
            header += "mfccs_std_" + str(i) + " "

        header += "cent_mean cent_std cent_skew "

        for i in range(0, 7):
            header += "contrast_mean_" + str(i) + " "
        for i in range(0, 7):
            header += "contrast_std_" + str(i) + " "

        header += "rolloff_mean rolloff_std rolloff_skew zrate_mean zrate_std zrate_skew tempo emotion"

        return header.split()

    def featuresExtractor(song_name = "", y = [], sr = 22500):
        
        if(song_name != ""): y, sr = librosa.load(song_name)
        
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        tempo, beat_frames = librosa.beat.beat_track(y=y_harmonic, sr=sr)
        tempo = int(tempo)

        chroma = librosa.feature.chroma_cens(y=y_harmonic, sr=sr)

        cent = librosa.feature.spectral_centroid(y=y, sr=sr)

        contrast = librosa.feature.spectral_contrast(y=y_harmonic, sr=sr)

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        zcr = librosa.feature.zero_crossing_rate(y_harmonic)

        mfccs = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=Extractor.__mfccs_n)

        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)

        chroma_df = pd.DataFrame()
        for i in range(0, 12):
            chroma_df["chroma_mean_" + str(i)] = chroma_mean[i]
        for i in range(0, 12):
            chroma_df["chroma_std_" + str(i)] = chroma_mean[i]
        
        chroma_df.loc[0] = np.concatenate((chroma_mean, chroma_std), axis=0)

        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)

        mfccs_df = pd.DataFrame()
        for i in range(0, Extractor.__mfccs_n):
            mfccs_df["mfccs_mean_" + str(i)] = mfccs_mean[i]
        for i in range(0, Extractor.__mfccs_n):
            mfccs_df["mfccs_std_" + str(i)] = mfccs_mean[i]
        mfccs_df.loc[0] = np.concatenate((mfccs_mean, mfccs_std), axis=0)

        cent_mean = np.mean(cent)
        cent_std = np.std(cent)
        cent_skew = scipy.stats.skew(cent, axis=1)[0]

        contrast_mean = np.mean(contrast, axis=1)
        contrast_std = np.std(contrast, axis=1)

        rolloff_mean = np.mean(rolloff)
        rolloff_std = np.std(rolloff)
        rolloff_skew = scipy.stats.skew(rolloff, axis=1)[0]

        spectral_df = pd.DataFrame()
        collist = ["cent_mean", "cent_std", "cent_skew"]
        for i in range(0, 7):
            collist.append("contrast_mean_" + str(i))
        for i in range(0, 7):
            collist.append("contrast_std_" + str(i))
        collist = collist + ["rolloff_mean", "rolloff_std", "rolloff_skew"]
        for c in collist:
            spectral_df[c] = 0
        data = np.concatenate(
                (
                    [cent_mean, cent_std, cent_skew],
                    contrast_mean,
                    contrast_std,
                    [rolloff_mean, rolloff_std, rolloff_skew],
                ),
                axis=0,
            )
        spectral_df.loc[0] = data

        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        zcr_skew = scipy.stats.skew(zcr, axis=1)[0]

        zrate_df = pd.DataFrame()
        zrate_df["zrate_mean"] = 0
        zrate_df["zrate_std"] = 0
        zrate_df["zrate_skew"] = 0
        zrate_df.loc[0] = [zcr_mean, zcr_std, zcr_skew]

        beat_df = pd.DataFrame()
        beat_df["tempo"] = tempo
        beat_df.loc[0] = tempo

        to_append = pd.concat(
                (chroma_df, mfccs_df, spectral_df, zrate_df, beat_df), axis=1
            ).to_string(header=False, index=False, index_names=False)

        return to_append.split()

    def __csvMaker(file_name, data, perms = "a"):
        file = open(file_name, perms, newline="")
        with file:
            writer = csv.writer(file)
            writer.writerow(data)
            file.close()

    def featuresExtractorWindowed(self, track, sr = 22050, window_seconds = 15, hop_seconds = 10):
        duration = int(librosa.get_duration(y=track, sr=sr))
        frame_len, hop_len = min(window_seconds, duration)*sr, min(hop_seconds, duration)*sr 
        frames = librosa.util.frame(
            track, frame_length=frame_len, hop_length=hop_len, axis=0)
        data = pd.DataFrame()
        for y in frames:
            
            data =  data.append(pd.Series(Extractor.featuresExtractor(y = y, sr = sr)), ignore_index = True)
        
        return np.array(data, dtype = float)

    def datasetMaker(self,
        path_tracks = "Tracks/",
        file_name="dataset",
        encoder=LabelEncoder(),
        sr = 22050,
        window_seconds = 15,
        hop_seconds = 10,
        print_metadata = False
    ):
        encoder = encoder.fit(Extractor.__emotions)
        
        Extractor.__csvMaker(file_name + ".csv", Extractor.__header, "w")

        if(print_metadata): 
            Extractor.__csvMaker(file_name= file_name + "_metadata.csv", data="file_name start end classID class".split(), perms="w")
        
        count = 0
        for g in Extractor.__emotions:
            for filename in os.listdir(f"{path_tracks}{g}"):
                
                songname = os.fspath(f"{path_tracks}{g}/{filename}")
                
                track, sr = librosa.load(songname, mono=True, sr = sr)

                count += 1
                print(str(count) + ")New track: " + songname)
                
                frame_len, hop_len = window_seconds*sr, hop_seconds*sr  # 15s window con 10s hop
                frames = librosa.util.frame(
                    track, frame_length=frame_len, hop_length=hop_len, axis=0
                )

                start = 0
                end = int(frame_len / sr)

                for y in frames:
                    
                    data = Extractor.featuresExtractor(y = y, sr = sr)
                    data.append(g)
                    Extractor.__csvMaker(file_name + ".csv", data)
                    
                    if(print_metadata):     
                        data = [
                            filename.encode("utf-8"),
                            start,
                            end,
                            encoder.transform([g])[0],
                            g,
                        ]

                        Extractor.__csvMaker(file_name + "_metadata.csv", data)
                        start += int(hop_len / sr)
                        end = start + int(frame_len / sr)