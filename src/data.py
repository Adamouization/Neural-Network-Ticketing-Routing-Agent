import numpy as np
import pandas as pd
from sklearn import preprocessing


class Data:
    """
    Data reader and encoder.
    """

    def __init__(self, csv_file_name):
        """
        Initialise data.
        :param csv_file_name:
        """
        # Get data.
        self.data_directory = "../data/"
        headers = self.parse_csv_headers(csv_file_name)
        raw_data = self.read_csv_file(csv_file_name, headers)

        # Separate input data and target data.
        self.input_data = raw_data.iloc[:, 0:raw_data.columns.size - 1]  # Data from all columns except last one.
        self.target_data = raw_data.iloc[:, -1]  # Data from last column
        self.target_data_encoded = list()

        # Data pre-processing encoders
        self.label_encoder = preprocessing.LabelEncoder()

    def read_csv_file(self, csv_file_name, headers):
        return pd.read_csv(
            "{}{}.csv".format(self.data_directory, csv_file_name),
            names=headers,
            skiprows=[0]
        )

    def encode_target_data(self):
        integer_encoded = self._integer_encode()
        binary_encoded = self._binary_encode(integer_encoded)
        one_hot_encoded = self._one_hot_encode(binary_encoded)
        self.target_data_encoded = one_hot_encoded

    def _integer_encode(self):
        return self.label_encoder.fit_transform(self.target_data)

    @staticmethod
    def _binary_encode(integer_encoded):
        return integer_encoded.reshape(len(integer_encoded), 1)

    @staticmethod
    def _one_hot_encode(binary_encode):
        one_hot_encoder = preprocessing.OneHotEncoder(sparse=False)
        onehot_encoded = one_hot_encoder.fit_transform(binary_encode)
        return onehot_encoded

    def inverse_encoding(self, onehot_encoded):
        # Invert first example
        inverted = self.label_encoder.inverse_transform([np.argmax(onehot_encoded[0, :])])
        print("Inverted encoding: {}".format(inverted))

    def parse_csv_headers(self, csv_file_name):
        file = pd.read_csv("{}{}.csv".format(self.data_directory, csv_file_name))
        return list(file.head(0))

    def print_input_data(self):
        print("Input data:")
        print(self.input_data)

    def print_target_data(self):
        print("Target data:")
        print(self.target_data)

    def print_encoded_target_data(self):
        print("One-hot encoded target data:")
        print(self.target_data_encoded)
