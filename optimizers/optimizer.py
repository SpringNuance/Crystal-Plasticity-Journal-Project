
from abc import ABC, abstractmethod

class optimizer(ABC):
    @abstractmethod
    def initializeOptimizer(self, *args, **kwargs):
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    @abstractmethod
    def outputResult(self, *args, **kwargs):
        pass


