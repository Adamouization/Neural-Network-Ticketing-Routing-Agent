import pandas as pd


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
        self.headers = self._parse_csv_headers(csv_file_name)
        raw_data = self._read_csv_file(csv_file_name)

        # Separate input data and target data.
        self.input_data = None
        self.input_data_encoded = list()
        self.target_data = None
        self.target_data_encoded = list()
        self._separate_input_target_data(raw_data)

    def _parse_csv_headers(self, csv_file_name):
        file = pd.read_csv("{}{}.csv".format(self.data_directory, csv_file_name))
        return list(file.head(0))

    def _read_csv_file(self, csv_file_name):
        return pd.read_csv(
            "{}{}.csv".format(self.data_directory, csv_file_name),
            names=self.headers,
            skiprows=[0]
        )

    def _separate_input_target_data(self, raw_data):
        self.input_data = raw_data.iloc[:, 0:raw_data.columns.size - 1]  # Data from all columns except last one.
        self.target_data = raw_data.iloc[:, -1]  # Data from last column

    def encode_input_data(self):
        """
        Encode input data with 0s or 1s for it to be readable by the MLPClassifier.
        :return: None
        """
        encoded_value = int()
        self.input_data_encoded = self.input_data.copy(deep=False)
        for index, row in self.input_data.iterrows():
            for i, v in row.iteritems():
                if v not in [0, 1]:
                    if v == "No":
                        encoded_value = 0
                    elif v == "Yes":
                        encoded_value = 1
                    else:
                        print("Error while encoding input data")
                        exit(1)
                    self.input_data_encoded.at[index, i] = encoded_value

    def encode_target_data(self):
        column = self.headers[-1]
        self.target_data_encoded = pd.get_dummies(self.target_data, columns=column)

    def print_input_data(self):
        print("Input data:")
        print(self.input_data)

    def print_target_data(self):
        print("Target data:")
        print(self.target_data)

    def print_encoded_target_data(self):
        print("One-hot encoded target data:")
        print(self.target_data_encoded)


def inverse_encoding(onehot_encoded):
    """
    Convert from one-hot encoding back to categorical string value.
    :param onehot_encoded: The one-hot encoded data to convert back.
    :return: The decoded data.
    """
    onehot_encoded = pd.DataFrame(onehot_encoded)
    return onehot_encoded.idxmax(axis=1)


def inverse_encoding_no_categories(onehot_encoded, categories):
    newlist = pd.Series()
    for i, p in enumerate(inverse_encoding(onehot_encoded)):
        newlist.at[i] = categories[p]
    return newlist
