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
        for_tsfedl=False,
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
        self.for_tsfedl = for_tsfedl
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
        if self.for_tsfedl:
            input_sequence = input_sequence.permute(1, 0)
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







# Seconds Time Series Dataset and acumulative data
class TimeSeriesDatasetSeconds(Dataset):
    def __init__(
        self,
        data,
        time_index=None,
        sequence_length=1,
        npred=1,
        for_forecasting=True,
        for_transformer=True,
        for_tsfedl=False,
        for_test=False,
        time_diff_threshold=60 * 30,
    ):
        """
        Custom PyTorch dataset for time series data.

        Parameters:
            data (torch.Tensor): The time series data, where each row represents a sequence of data.
            sequence_length (int): The length of input sequences.
            npred (int): Number of steps to forecast. Will be placed as 0 if for_forecasting is False.
            for_forecasting (bool): Set to True for forecasting, False for autoencoder.
        """
        self.time_index = time_index
        self.time_diff = np.append([0], (np.diff(time_index) / 1e9).astype(float))
        self.data = data
        self.sequence_length = sequence_length
        self.npred = npred if for_forecasting else 0
        self.for_forecasting = for_forecasting
        self.for_transformer = for_transformer
        self.time_diff_threshold = time_diff_threshold
        self.for_tsfedl = for_tsfedl
        self.for_test = for_test

        self.time_diff_mask = self.time_diff > self.time_diff_threshold

        len_ = len(self.data) - self.sequence_length - self.npred
        cumulative_sum = np.cumsum(self.time_diff_mask)
        cumulative_sum = np.pad(cumulative_sum, (1, 0), constant_values=0)
        window_size = self.sequence_length + self.npred
        window_sums = cumulative_sum[window_size:] - cumulative_sum[:-window_size]
        valid_start_indices = np.where(window_sums == 0)[0]
        self.valid_indexes_slices = []
        for i in valid_start_indices:
            input_indices = (i, i + self.sequence_length)
            target_indices = (
                (i + self.sequence_length, i + self.sequence_length + self.npred)
                if self.for_forecasting
                else input_indices
            )
            self.valid_indexes_slices.append((input_indices, target_indices))

        self.length = len(self.valid_indexes_slices)
        print("Discarded {} instances from {}".format(len_ - self.length, len_))
        self.skip_instances = 0

    def __len__(self):
        """
        Returns:
            int: Number of sequences in the data.
        """
        return self.length

    def __getitem__(self, idx):
        """
        Returns:
            tuple: A tuple where the first element is the input sequence and the second element is the target sequence.
        """
        input_sequence = self.data[
            self.valid_indexes_slices[idx][0][0] : self.valid_indexes_slices[idx][0][1]
        ].double()
        if self.for_tsfedl:
            # Rotate -1 and -2 dim
            input_sequence = input_sequence.permute(1, 0)
        tgt_sequence = self.data[
            self.valid_indexes_slices[idx][1][0] : self.valid_indexes_slices[idx][1][1]
        ].double()
        timestamp_input, timestamp_tgt = None, None
        if self.for_test:
            timestamp_input = (
                (
                    self.time_index[
                        self.valid_indexes_slices[idx][0][
                            0
                        ] : self.valid_indexes_slices[idx][0][1]
                    ]
                    - np.datetime64("1970-01-01 00:00:00")
                )
                / 1e9
            ).astype("int64")
            timestamp_tgt = (
                (
                    self.time_index[
                        self.valid_indexes_slices[idx][1][
                            0
                        ] : self.valid_indexes_slices[idx][1][1]
                    ]
                    - np.datetime64("1970-01-01 00:00:00")
                )
                / 1e9
            ).astype("int64")
        if self.for_transformer:
            if self.for_test:
                return (
                    input_sequence,
                    tgt_sequence,
                    tgt_sequence,
                    timestamp_input,
                    timestamp_tgt,
                    self.valid_indexes_slices[idx],
                )
            else:
                return input_sequence, tgt_sequence, tgt_sequence
        else:
            if self.for_test:
                return (
                    input_sequence,
                    tgt_sequence,
                    timestamp_input,
                    timestamp_tgt,
                    self.valid_indexes_slices[idx],
                )
            else:
                return input_sequence, tgt_sequence


if __name__ == "__main__":
    # Example Usage:
    # Create a time series dataset for forecasting
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    sequence_length = 3
    npred = 2
    dataset = TimeSeriesDatasetSeconds(data, sequence_length, npred, for_forecasting=True)

    # Create a time series dataset for autoencoder
    dataset_autoencoder = TimeSeriesDatasetSeconds(
        data, sequence_length, npred, for_forecasting=False
    )
    print("First batch")
    print(dataset[0])
    print(dataset_autoencoder[0])
    print("Second batch")
    print(dataset[1])
    print(dataset_autoencoder[1])
