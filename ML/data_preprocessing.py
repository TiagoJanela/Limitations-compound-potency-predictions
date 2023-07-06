# Imports
import numpy as np
import random


class Dataset:
    def __init__(self, features: np.array, labels: np.array):
        self.features = features
        self.labels = labels
        self._add_instances = set()

    def add_instance(self, name, values: np.array):
        self._add_instances.add(name)
        self.__dict__[name] = values

    @property
    def columns(self) -> dict:
        data_dict = {k: v for k, v in self.__dict__.items()}
        data_dict['features'] = self.features
        data_dict['labels'] = self.labels
        return data_dict

    def __len__(self):
        return self.labels.shape[0]

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return {col: values[idx] for col, values in self.columns.items()}

        subset = Dataset(self.features[idx], self.labels[idx])
        for addt_instance in self._add_instances:
            subset.add_instance(addt_instance, self.__dict__[addt_instance][idx])

        return subset

    def shuffle(self):
        random.Random(1).shuffle(self.labels)
