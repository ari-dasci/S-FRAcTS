from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        data,
        time_index=None,
        sequence_length=1,
        npred=1,
        for_forecasting=True,
        for_transformer=True,
        for_test=False,
        use_scaler=True,
        time_unit="m",  # "h" for hours, "m" for minutes
        freq=15,  # 1 min or 15 min
        scaler=None,  # pre-fitted scaler (e.g. from training data); if None, fit on `data`
    ):
        """
        Custom PyTorch dataset for time series data with optional scaling.
        """
        self.time_index = time_index
        self.sequence_length = sequence_length
        self.npred = npred if for_forecasting else 0
        self.for_forecasting = for_forecasting
        self.for_transformer = for_transformer
        self.for_test = for_test
        self.use_scaler = use_scaler
        self.time_unit = time_unit
        self.freq = freq

        if self.use_scaler:
            if scaler is None:
                self.scaler = StandardScaler()
                data = self.scaler.fit_transform(data)
            else:
                self.scaler = scaler
                data = self.scaler.transform(data)
        
        self.data = torch.tensor(data, dtype=torch.float32)

        len_ = len(self.data) - self.sequence_length - self.npred
        self.valid_indexes_slices = [
            (
                (i, i + self.sequence_length),
                ((i + self.sequence_length, i + self.sequence_length + self.npred)
                 if self.for_forecasting else (i, i + self.sequence_length))
            )
            for i in range(len_)
        ]
        self.length = len(self.valid_indexes_slices)
        print(f"Using {self.length} valid sequences.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_sequence = self.data[
            self.valid_indexes_slices[idx][0][0] : self.valid_indexes_slices[idx][0][1]
        ].double()
        tgt_sequence = self.data[
            self.valid_indexes_slices[idx][1][0] : self.valid_indexes_slices[idx][1][1]
        ].double()
        
        timestamp_input, timestamp_tgt = None, None
        if self.for_test and self.time_index is not None:
            time_scale = np.timedelta64(self.freq, self.time_unit)
            timestamp_input = (
                (
                    self.time_index[
                        self.valid_indexes_slices[idx][0][0] : self.valid_indexes_slices[idx][0][1]
                    ] - np.datetime64("1970-01-01 00:00:00")
                ) / time_scale
            ).astype("int64")
            timestamp_tgt = (
                (
                    self.time_index[
                        self.valid_indexes_slices[idx][1][0] : self.valid_indexes_slices[idx][1][1]
                    ] - np.datetime64("1970-01-01 00:00:00")
                ) / time_scale
            ).astype("int64")
        
        if self.for_transformer:
            return (input_sequence, tgt_sequence, tgt_sequence, timestamp_input, timestamp_tgt, self.valid_indexes_slices[idx]) if self.for_test else (input_sequence, tgt_sequence, tgt_sequence)
        else:
            return (input_sequence, tgt_sequence, timestamp_input, timestamp_tgt, self.valid_indexes_slices[idx]) if self.for_test else (input_sequence, tgt_sequence)

    def inverse_transform(self, data):
        """Inverse transform the scaled data back to the original scale."""
        if self.use_scaler:
            return self.scaler.inverse_transform(data)
        return data
