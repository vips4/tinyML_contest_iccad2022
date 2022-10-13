import os
from utils.data import loadCSV, txt_to_numpy


class IEGM_DataSET:
    def __init__(
        self,
        root_dir: str,
        indice_dir: str,
        mode: str,  # Literal["train", "test"]
        size: int = 1250,
    ):
        self.root_dir = root_dir
        self.indice_dir = indice_dir
        self.size = size
        self.names_list = []

        csvdata_all = loadCSV(os.path.join(self.indice_dir, mode + "_indexes.csv"))

        for i, (k, v) in enumerate(csvdata_all.items()):
            self.names_list.append(str(k) + " " + str(v[0]))

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        text_path = self.root_dir + self.names_list[idx].split(" ")[0]

        if not os.path.isfile(text_path):
            print(text_path + "does not exist")
            return None

        IEGM_seg = txt_to_numpy(text_path, self.size).reshape(1, self.size, 1)
        label = int(self.names_list[idx].split(" ")[1])
        sample = {"IEGM_seg": IEGM_seg, "label": label}

        return sample


if __name__ == "__main__":
    d = IEGM_DataSET("./data/data_training/", "./data/", "train", 1250)
