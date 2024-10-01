from .base_model import BaseModel
class LIF(BaseModel):
    def __init__(self):
        pass
    
    def update(self, input_signal):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def reset(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
