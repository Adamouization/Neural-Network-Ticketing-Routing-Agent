import pandas as pd


class DataProcessor:
    """
    Data processor, which reads, splits and encodes data to be used by SciKit's MLPClassifier.
    """

    def __init__(self, csv_file_name):
        """
        Initialise variables and the data.
        :param csv_file_name: The name of CSV file containing the input and target data.
        """
        # Get the data from the CSV file.
        self.data_directory = "../data/"
        self.headers = self._parse_csv_headers(csv_file_name)
        raw_data = self._read_csv_file(csv_file_name)

        # Separate input data from target data.
        self.input_data = None
        self.input_data_encoded = list()
        self.target_data = None
        self.target_data_encoded = list()
        self._separate_input_target_data(raw_data)

    def _parse_csv_headers(self, csv_file_name):
        """
        Retrieves the headers from the CSV file.
        :param csv_file_name: The name of CSV file containing the input and target data.
        :return: A list of strings containing the CSV headers.
        """
        file = pd.read_csv("{}{}.csv".format(self.data_directory, csv_file_name))
        return list(file.head(0))

    def _read_csv_file(self, csv_file_name):
        """
        Reads the CSV file and stores its data in a Pandas DataFrame.
        :param csv_file_name: The name of CSV file containing the input and target data.
        :return: A DataFrame containing the data from the CSV file.
        """
        return pd.read_csv(
            "{}{}.csv".format(self.data_directory, csv_file_name),
            names=self.headers,
            skiprows=[0]
        )

    def _separate_input_target_data(self, raw_data):
        """
        Separates the input data from target data. All data contained in the last column corresponds to the target data,
        while the rest corresponds to the input data.
        :param raw_data: The DataFrame containing all data read from the CSV file.
        :return: None
        """
        self.input_data = raw_data.iloc[:, 0:raw_data.columns.size - 1]  # Data from all columns except last one.
        self.target_data = raw_data.iloc[:, -1]  # Data from last column

    def encode_input_data(self):
        """
        Encode input data with in integer format (0s or 1s) for it to be readable by SciKit's MLPClassifier.
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
        """
        One-hot encode the target data for it to be readable by SciKit's MLPClassifier.
        :return: None
        """
        column = self.headers[-1]
        self.target_data_encoded = pd.get_dummies(self.target_data, columns=column)


def inverse_encoding(onehot_encoded):
    """
    Convert from one-hot encoding back to categorical string value.
    :param onehot_encoded: The one-hot encoded data to convert back.
    :return: The decoded data.
    """
    onehot_encoded = pd.DataFrame(onehot_encoded)
    return onehot_encoded.idxmax(axis=1)


def inverse_encoding_no_categories(onehot_encoded, categories):
    """
    Convert from one-hot encoding back to categorical string value including the category names rather than just
    indexes.
    :param onehot_encoded: The one-hot encoded data to convert back (with indexes as categories).
    :param categories: The string categories.
    :return: The decoded data in the form of Pandas series.
    """
    not_encoded = pd.Series()
    for i, p in enumerate(inverse_encoding(onehot_encoded)):
        not_encoded.at[i] = categories[p]
    return not_encoded
