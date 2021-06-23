import librosa
import numpy as np
import scipy.linalg
import scipy.stats
import warnings

warnings.filterwarnings(action="ignore")


class Song:
    MAJOR_PROFILE = [
        6.35,
        2.23,
        3.48,
        2.33,
        4.38,
        4.09,
        2.52,
        5.19,
        2.39,
        3.66,
        2.29,
        2.88,
    ]
    MINOR_PROFILE = [
        6.33,
        2.68,
        3.52,
        5.38,
        2.60,
        3.53,
        2.54,
        4.75,
        3.98,
        2.69,
        3.34,
        3.17,
    ]
    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def __init__(self, audio_path):
        self.audio_time_series, self.audio_sr = librosa.load(audio_path)

    def get_pitch_class_distribution(self):
        """
        Also known as Chroma Vector.
        """
        chromagram = librosa.feature.chroma_stft(
            y=self.audio_time_series, sr=self.audio_sr
        )
        return np.sum(chromagram, axis=1)

    def get_tempo(self):
        """
        If the tempo varies, it gets the mean.
        """
        return np.mean(librosa.beat.tempo(self.audio_time_series, sample_rate))

    def _return_zscore(self, array):
        return scipy.stats.zscore(array)

    def get_estimated_song_key(self):
        """
        Uses the Krumhansl-Schmuckler key-finding algorithm to estimate a song's key, given a pitch class distribution.
        More info: http://rnhart.net/articles/key-finding/
        what in the nani heck is gg on
        """
        # X is a chroma vector, a numpy ndarray of shape (12,)
        # That is normalized with ZScore
        X = self._return_zscore(self.get_pitch_class_distribution())

        # shape: (12,)
        major_profile_zscore = self._return_zscore(self.MAJOR_PROFILE)
        minor_profile_zscore = self._return_zscore(self.MINOR_PROFILE)

        # shape: (12, 12)
        major_profile_rotated = scipy.linalg.circulant(major_profile_zscore)
        minor_profile_rotated = scipy.linalg.circulant(minor_profile_zscore)

        # chroma vector . coefficients
        major_X = major_profile_rotated.T.dot(X)
        minor_X = minor_profile_rotated.T.dot(X)
        print(major_X, minor_X)

        major_winner = np.argmax(major_X)
        minor_winner = np.argmax(minor_X)

        if major_X[major_winner] > minor_X[minor_winner]:
            return f"{self.NOTE_NAMES[major_winner]} major"
        elif major_X[major_winner] < minor_X[minor_winner]:
            return f"{self.NOTE_NAMES[minor_winner]} minor"
        else:
            return f"{self.NOTE_NAMES[major_winner]} major or {self.NOTE_NAMES[minor_winner]} minor"


if __name__ == "__main__":
    audio_filepath = "./test_data/sam_smith.mp3"
    print(audio_filepath)
    song = Song(audio_filepath)
    print(song.get_estimated_song_key())

    audio_filepath = "./test_data/zimzalabim.mp3"
    print(audio_filepath)
    song = Song(audio_filepath)
    print(song.get_estimated_song_key())
