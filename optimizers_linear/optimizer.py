
from abc import ABC, abstractmethod

class optimizer(ABC):
    @abstractmethod
    def InitializeFirstStageOptimizer(self, *args, **kwargs):
        pass

    @abstractmethod
    def FirstStageRun(self, *args, **kwargs):
        pass

    @abstractmethod
    def FirstStageOutputResult(self, *args, **kwargs):
        pass

    @abstractmethod
    def InitializeSecondStageOptimizer(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def SecondStageRun(self, *args, **kwargs):
        pass

    @abstractmethod
    def SecondStageOutputResult(self, *args, **kwargs):
        pass


