from abc import ABC, abstractmethod


class Segmenter(ABC):
    @abstractmethod
    def segment(self, data, **kwargs):
        pass
