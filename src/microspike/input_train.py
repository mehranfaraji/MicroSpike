class InputTrain():
    def __init__(self,
                spikes_t,
                spikes_i,
                position_copypaste=None) -> None:
        self.spikes_t = spikes_t
        self.spikes_i = spikes_i
        self.position_copypaste = position_copypaste