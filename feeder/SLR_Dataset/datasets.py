import torch
import torch.utils.data

from feeder.single_dataset.ISLGoaNew import ISL_GOA
from feeder.single_dataset.NMFs_CSL import NMFs_CSL
from feeder.single_dataset.MSASL import MSASL
from feeder.single_dataset.WLASL import WLASL
from feeder.single_dataset.SLR500 import SLR500



class TotalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_split='train',
        data_root='/data',
        subset_name=('SLR500', 'MS_ASL', 'WLASL', 'NMFs_CSL'),
        frames=65,
        threshold=0.4,
        interval=2,
        hand_side='right',
        msasl_class_num=1000,
        wlasl_class_num=2000,
        ds_ratio=1.0,
        isl_root='/home/nithin/Desktop/ISL_Goa_Data/MASA/Data/ISL_GOA',
        isl_list_file='/home/nithin/Desktop/ISL_Goa_Data/MASA/Data/ISL_GOA/Annotations/pretrain_list.txt'
    ):
        """
        TotalDataset merges multiple datasets.
        Each dataset class must provide:
            - __len__()
            - get_sample(index)
        """
        self.data_split = data_split
        self.data_root = data_root
        self.frames = frames
        self.threshold = threshold
        self.interval = interval
        self.hand_side = hand_side
        self.ds_ratio = float(ds_ratio)

        # aug is set to False by default; set to True if augmentation is implemented
        self.aug = False

        # Build dataset map
        dataset_to_index = {}

	# ---- ISLGoaNew ----
        if 'ISLGoaNew' in subset_name:
            self.ISL_GOA = ISL_GOA(
                data_root=isl_root,
                list_file=isl_list_file,
                split=data_split,
                pose_subdir='Pose'
            )
            print(f"Initialized ISL_GOA with {len(self.ISL_GOA)} samples")
            dataset_to_index['ISLGoaNew'] = self.ISL_GOA

        # ---- SLR500 ----
        if 'SLR500' in subset_name:
            self.SLR500 = SLR500(
                data_split=self.data_split,
                interval=self.interval,
                threshold=self.threshold,
                augment=self.aug
            )
            print(self.SLR500)
            dataset_to_index['SLR500'] = self.SLR500

        # ---- NMFs_CSL ----
        if 'NMFs_CSL' in subset_name:
            self.NMFs = NMFs_CSL(
                data_split=self.data_split,
                interval=self.interval,
            )
            print(self.NMFs)
            dataset_to_index['NMFs_CSL'] = self.NMFs

        # ---- MS_ASL ----
        if 'MS_ASL' in subset_name:
            self.MS_ASL = MSASL(
                data_split=self.data_split,
                interval=self.interval,
                class_num=msasl_class_num,
            )
            print(self.MS_ASL)
            dataset_to_index['MS_ASL'] = self.MS_ASL

        # ---- WLASL ----
        if 'WLASL' in subset_name:
            self.WLASL = WLASL(
                data_split=self.data_split,
                interval=self.interval,
                subset_num=wlasl_class_num,
            )
            print(self.WLASL)
            dataset_to_index['WLASL'] = self.WLASL

        # ---- ISL_GOA ----
        if 'ISL_GOA' in subset_name:
            self.ISL_GOA = ISL_GOA(
                data_root=isl_root,
                split=self.data_split,
                list_file=isl_list_file
            )
            print(self.ISL_GOA)
            dataset_to_index['ISL_GOA'] = self.ISL_GOA

        # Keep the order requested by subset_name, only include those actually loaded
        self.datasets = []
        for name in subset_name:
            if name not in dataset_to_index:
                raise ValueError(
                    f"Dataset '{name}' requested in subset_name but not loaded. "
                    f"Check spelling or whether its loader is defined."
                )
            self.datasets.append(dataset_to_index[name])

        # Total size (with ratio)
        self.total_data = 0
        for ds in self.datasets:
            self.total_data += int(len(ds) * self.ds_ratio)

    def __len__(self):
        return self.total_data

    def len(self):
        return self.__len__()

    def __getitem__(self, index):
        # DataLoader calls __getitem__; internally routes to get_sample
        return self.get_sample(index)

    def get_sample(self, index):
        sample, _ = self._get_sample(index)
        return sample

    def _get_sample(self, index):
        base = 0
        for ds in self.datasets:
            ds_len = int(len(ds) * self.ds_ratio)
            if index < base + ds_len:
                # Map global index to per-dataset index
                sample = ds.get_sample(index - base)
                return sample, ds
            base += ds_len

        # If index is out of range
        raise IndexError(
            f"Index {index} out of range for total length {self.total_data}"
        )
