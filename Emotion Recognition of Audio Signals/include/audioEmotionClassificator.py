from asyncio.windows_events import NULL
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
import pandas as pd
import numpy as np
import seaborn as sn

from include import audioFeatures as af
from IPython.display import Audio
from IPython.display import display

import librosa

import librosa.display

from datetime import timedelta

import pyaudio
import wave


class ERAS(object):
    def __init__(self, dataset = "dataset(mffc=40).csv", emotions = "arrabbiato calmo cupo drammatico felice motivante romantico triste vivace", mfccs_n = 40):
        ERAS.__model = None
        ERAS.__scaler = None
        ERAS.__encoder = None
        ERAS.__dataset = dataset
        ERAS.__extractor = af.Extractor(emotions, mfccs_n)
        ERAS.__colors = {'Angry': 'red',
                'Inspirational': 'lightblue',
                'Sad': 'gray',
                'Dark': 'black',
                'Happy': 'yellow',
                'Bright': 'orange',
                'Calm': 'green',
                'Romantic': 'pink',
                'Dramatic': 'purple'}
        ERAS.__categories = "Angry Calm Dark Dramatic Happy Inspirational Romantic Sad Bright".split()

    def setDataset(dataset): ERAS.__dataset = dataset
    
    def __dataPreProcessing(scaler, test_size = 0.2):
        
        data = pd.read_csv(ERAS.__dataset, engine='python')
        data.head()
        
        emotion_list = data.iloc[:, -1]
        encoder = LabelEncoder()
        encoder = encoder.fit(ERAS.__categories)
        y = encoder.transform(emotion_list)
        
        X = np.array(data.iloc[:, :-1], dtype = float)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = test_size)
        
        scaler = scaler.fit(X_train)
        
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test, scaler, encoder

    def __modelScore(model, X_test, y_test):
            
        prediction = model.predict(X_test)

        print(classification_report(prediction, y_test, target_names = ERAS.__categories))
        df_cm = pd.DataFrame(confusion_matrix(prediction, y_test, normalize='true'), index = [i for i in ERAS.__categories],
                            columns = [i for i in ERAS.__categories])
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True)

        plt.show()

    def train(self, test_size = 0.2, model = KNeighborsClassifier(n_neighbors=2), showScore = True, scaler=StandardScaler()):
        
        
        X_train, X_test, y_train, y_test, scaler, encoder = ERAS.__dataPreProcessing(scaler, test_size)
        
        strtfdKFold = StratifiedKFold(n_splits=10)
        kfold = strtfdKFold.split(X_train, y_train)
        scores = []
        
        for k, (train, test) in enumerate(kfold):
            model.fit(X_train[train], y_train[train])
            score = model.score(X_train[test], y_train[test])
            scores.append(score)
            print('Fold: %2d, Training/Test Split Distribution: %s, Accuracy: %.3f' % (k+1, np.bincount(y_train[train]), score))
    
        print('\n\nCross-Validation accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

        #model = model.fit(X_train, y_train)
        
        if(showScore):
           ERAS.__modelScore(model, X_test, y_test)
        
        
        ERAS.__model = model
        ERAS.__encoder = encoder
        ERAS.__scaler = scaler
        return self

    def predict(self, track, offset = None, duration = None, sr = 22050, window_seconds = 15, hop_seconds = 10, transcribe = False):
        
        if(ERAS.__model == None):
            print("Model is not trained")
            return
        
        y, sr = librosa.load(track, offset = offset, duration = duration, mono = True, sr = sr)

        features = ERAS.__extractor.featuresExtractorWindowed(track = y, sr = sr, window_seconds = window_seconds, hop_seconds = hop_seconds)

        prediction = ERAS.__encoder.inverse_transform(ERAS.__model.predict(ERAS.__scaler.transform(features)))
        
        ERAS.__printResult(prediction, window_seconds, hop_seconds)

        if(transcribe):
            ERAS.__transcribe(y)

        full = Audio(y,rate = sr)

        display(full)
        
    def __transcribe(y):
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        import whisper

        whisper_model = whisper.load_model("small")
        whisper_model.cuda(0)
        result = whisper_model.transcribe(y)
        print("Language: " + result['language'])
        print(result["text"])
        sid = SentimentIntensityAnalyzer()
        print(sid.polarity_scores(result["text"]))
        
    def __printResult(prediction, window_seconds, hop_seconds):
        prediction = pd.DataFrame(prediction)
        
        df = pd.DataFrame(columns=['FrameTime start', 'FrameTime End', 'Emotions'])
    
        df['Emotions'] = prediction
        StartA = []
        EndA = []
        start = 0
        for i in range(0,df.shape[0]):
            StartA.append(start)
            EndA.append(start+window_seconds)
            start += hop_seconds
        df['FrameTime start'] = StartA;
        df['FrameTime End'] = EndA

        fig, ax = plt.subplots(ncols=2, figsize=(18,8))
        prediction = df['Emotions'].value_counts().sort_values(ascending=False);
        df['Emotions'].value_counts().plot.pie(labels = prediction.index, colors=[ERAS.__colors[key] for key in prediction.index], ax=ax[0])

        ax[0].set_title('Emotions pie plot')
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Emotion")
        ax[1].plot(df['FrameTime start'],df['Emotions'])
        
        plt.show()

        fig = plt.figure(figsize=(20,6))
        ax = fig.add_subplot(111)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_aspect(1)
        
        times = np.arange(0, max(df['FrameTime End'][:]), hop_seconds)
    
        labels = [str(timedelta(seconds=int(seconds)))[2:] for seconds in times]
        plt.xticks(times, labels)
        plt.xlabel('Time')
        plt.axis()
        for row in df.iterrows():
            x = row[1]['FrameTime start']
            x1 = x + hop_seconds
            plt.fill_between([x,x1], 1,y2=15, color=ERAS.__colors[row[1]['Emotions']])
            plt.text((x+x1)/2,1,row[1]['Emotions'], horizontalalignment='center', verticalalignment='bottom')

        plt.ylim(max(3,max(df['FrameTime End'][:])*3/100),1)
        plt.show()
        
    def voiceEmotion(self, sr = 44100, seconds = 30):

        chunk = 1024  # Record in chunks of 1024 samples
        sample_format = pyaudio.paInt16  # 16 bits per sample
        channels = 2
        
        filename = "Validation/output.wav"

        p = pyaudio.PyAudio()  # Create an interface to PortAudio

        print('Recording')

        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=sr,
                        frames_per_buffer=chunk,
                        input=True)

        frames = []  # Initialize array to store frames

        # Store data in chunks for 3 seconds
        for i in range(0, int(sr / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)

        # Stop and close the stream 
        stream.stop_stream()
        stream.close()
        # Terminate the PortAudio interface
        p.terminate()

        print('Finished recording')

        # Save the recorded data as a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(sr)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        ERAS.predict(self,track=filename)