import librosa
import numpy as np
from math import ceil
from statistics import mean

class TempsAnalyzer:
    def __init__(self, chunk_length, lower_coof, upper_coof):
        self.chunk_length = chunk_length
        self.lower_cof = lower_coof
        self.upper_cof = upper_coof

        self.mean_temp = 0
        self.lower_border = 0
        self.upper_border = 0

        self.bradial = False
        self.tahial = False

        self.fragments_param = []
        self.results = {
            "fragments": []
        }

    # Делит аудио файл(фрагменты) на чанки
    def chunkizer(self, audio, sr):
        num_chunks = ceil(librosa.get_duration(audio, sr=sr) / self.chunk_length)
        chunks = []
        for i in range(num_chunks):
            chunks.append(audio[i * self.chunk_length * sr:(i + 1) * self.chunk_length * sr])
        return chunks

    # Считает средний темп для всех фрагиентов
    def calculate_mean_temp(self):
        for temp in self.fragments_param:
            if isinstance(temp[1], list):
                self.mean_temp += max(temp[1])
            else:
                self.mean_temp += temp[1]
        self.mean_temp /= len(self.fragments_param)

    def update_borders(self):
        self.lower_border = self.mean_temp - self.mean_temp * self.lower_cof
        self.upper_border = self.mean_temp + self.mean_temp * self.upper_cof

    def get_speech_frenzy(self, audio):
        f, _, _ = librosa.pyin(audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
        return f

    # Разделяет речь по паузам
    def split_to_lists(self, frenzy):
        last_val = frenzy[0]
        split_val = []
        cur_lst = []

        for val in frenzy:
            if np.isnan(val) != np.isnan(last_val):
                last_val = val
                split_val.append(cur_lst)
                cur_lst = []
            cur_lst.append(val)

        return split_val

    def count_abrupt_noise(self, splited_frenzy):
        """
        Находит отрывистые звуки
        Счётчик увеличивается при условии что предыдущие и следующие значение это пауза,
        а так же что их длинна не превышает 20
        """
        count = 0
        for i in range(1, len(splited_frenzy) - 1):
            if np.isnan(splited_frenzy[i - 1][0]) and np.isnan(splited_frenzy[i + 1][0]):
                if len(splited_frenzy[i - 1]) >= 20 and len(splited_frenzy[i]) >= 20:
                    count += 1
        return count

    # по частоте речи находит отрывистые звуки и запинания
    def work_with_frenzy(self, audio):
        f = self.get_speech_frenzy(audio)
        splited_f = self.split_to_lists(f)
        abrupt_noise = self.count_abrupt_noise(splited_f)
        return abrupt_noise

    def analyze_fragment(self, path_to_audio):
        fragments_temp = []

        self.load_audio(path_to_audio)
        self.get_duration()
        noise = self.work_with_frenzy(self.y)
        if self.duration > self.chunk_length:
            chunks = self.chunkizer(self.y, self.sr)
            for chunk in chunks:
                fragments_temp.append(self.get_temp(chunk))

            self.fragments_param.append([path_to_audio, fragments_temp, noise])

        else:
            temp = self.get_temp(self.y)

            self.fragments_param.append([path_to_audio, temp, noise])

    def load_audio(self, path_to_audio):
        self.y, self.sr = librosa.load(path_to_audio)

    def get_duration(self):
        self.duration = librosa.get_duration(self.y, self.sr)

    def get_temp(self, fragment):
        onset_env = librosa.onset.onset_strength(fragment, sr=self.sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=self.sr)
        return round(tempo[0])

    def check_lower_border(self, temp):
        if temp < self.lower_border:
            self.bradial = True

    def check_upper_border(self, temp):
        if temp > self.upper_border:
            self.tahial = True

    def reset_changes(self):
        self.bradial = False
        self.tahial = False

    def get_results(self):
        self.results["key_values"] = {"mean_temp": self.mean_temp,
                                       "lower_border": self.lower_border,
                                       "upper_border": self.upper_border}

        for temp in self.fragments_param:
            if isinstance(temp[1], list):
                changes = []
                self.check_lower_border(min(temp[1]))
                self.check_upper_border(max(temp[1]))
                for tm, tp in zip(temp[1], temp[1][1:]):
                    if tp > tm:
                        changes.append("Temp up")
                    elif tp < tm:
                        changes.append("Temp slow")
                    else:
                        changes.append("Temp don`t change")
                self.results["fragments"].append({"file_name": temp[0],
                                                  "temps": temp[1],
                                                  "mean_temp": round(mean(temp[1]), 1),
                                                  "significant_decrease": self.bradial,
                                                  "significant_increase": self.tahial,
                                                  "temp_change": changes,
                                                  "abrupt_noise": temp[2]})
                self.reset_changes()

            else:
                self.check_lower_border(temp[1])
                self.check_upper_border(temp[1])
                self.results["fragments"].append({"file_name": temp[0],
                                                  "temps": temp[1],
                                                  "mean_temp": temp[1],
                                                  "significant_decrease": self.bradial,
                                                  "significant_increase": self.tahial,
                                                  "temp_change": "Fragment is too short",
                                                  "abrupt_noise": temp[2]})
                self.reset_changes()
        return self.results
