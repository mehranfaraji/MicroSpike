import numpy as np
from numba import jit
import warnings

@jit(nopython=True)
def make_single_train(min_rate, max_rate, max_time_wo_spike, max_change_speed, runduration, dt, random_seed):
    np.random.seed(int(random_seed))
    runduration1 = min(runduration, 150) # [second]
    st = []
    virtual_pre_sim_spike = - np.random.rand() * max_time_wo_spike # in [-0.05, 0]
    firing_rate = min_rate + np.random.rand() * (max_rate - min_rate) # in [0, 90]
    rate_change = 2 * (np.random.rand() - 0.5) * max_change_speed # in [-1800,1800]

    mtws = max_time_wo_spike # 0.05 [seocnds]

    # for the third line of the condition, if the neuron has not spiked
    # for the last 50 ms or 0.050 seconds then it will definitely will spike

    for t in np.arange(dt, runduration1, dt):
        if np.random.rand() < dt * firing_rate or \
        (len(st) < 1 and t - virtual_pre_sim_spike > mtws) or \
        (len(st) > 0 and t - st[-1] > mtws):
            if t < 0 or t > runduration1:
                raise ValueError(f'tmp = {t} (tmp<0 or tmp>{runduration1} violated)')
            # t = max(0, t)
            # t = min(runduration1, t)
            t = np.round(t, decimals=3)
            st. append(t)
        firing_rate = firing_rate + rate_change * dt
        rate_change = rate_change + 1/5 * 2 * (np.random.rand() - 0.5) * max_change_speed
        rate_change = max(min(rate_change,max_change_speed), -max_change_speed)
        firing_rate = max(min(firing_rate, max_rate), min_rate)
    return st


class PatternGenerator:
    def __init__(self, number_pattern, runduration=450, tripling=True, dt=0.001, number_neurons=2000, portion_involved_pattern=0.5, patternlength=0.05,
                 max_rate=90, min_rate=0, max_time_wo_spike=0.05, max_change_speed=None, total_pattern_freq=1/3,):
        self.runduration = runduration
        self.tripling = tripling
        self.dt = dt
        self.number_neurons = number_neurons
        self.portion_involved_pattern = portion_involved_pattern
        self.patternlength = patternlength
        self.max_rate = max_rate
        self.min_rate = min_rate
        self.max_time_wo_spike = max_time_wo_spike
        if max_change_speed is None:
            self.max_change_speed = max_rate / max_time_wo_spike
        else:
            self.max_change_speed = max_change_speed
            if self.max_change_speed != max_rate / max_time_wo_spike:
                warnings.warn("Warning: max_change_speed != max_rate / max_time_wo_spike", UserWarning)
        self.total_pattern_freq = total_pattern_freq
        self.number_pattern = number_pattern
    def make_input(self):
        spiketimes, indices = [], []
        for n in range(self.number_neurons): 
            st = np.array(make_single_train(self.min_rate, self.max_rate, self.max_time_wo_spike,
                                            self.max_change_speed, self.runduration, self.dt,
                                            np.random.randint(2**30)))
            spiketimes.append(st)
            indices.append(n * np.ones(len(st)))

        spiketimes = np.hstack(spiketimes)
        indices = np.hstack(indices)
        sortarray = np.argsort(spiketimes)
        spiketimes = spiketimes[sortarray]
        indices = indices[sortarray]
        indices = indices.astype(int)
        return spiketimes, indices


    def make_pattern_presentation_array(self):
        runduration1 = min(self.runduration, 150)
        if self.total_pattern_freq == 0.5 and self.number_pattern == 1:
            position_copypaste = np.array([0,1] * int(runduration1 * self.total_pattern_freq / self.patternlength), dtype=int)
        else:
            position_copypaste = np.zeros(int(runduration1 / self.patternlength), dtype=int) # shape = (3000,)
            for pattern_i in range(1, self.number_pattern + 1):
                # while sum(...) < 250 * (pattern_i+1)*pattern_i/2
                # while sum(...) < 250 * (1+1)*1/2 = 250 * 1
                # while sum(...) < 250 * (2+1)*2/2 = 250 * 3
                # while sum(...) < 250 * (4+1)*4/2 = 250 * 10
                while sum(position_copypaste) < np.floor(int(runduration1 / self.patternlength) * self.total_pattern_freq / self.number_pattern * (pattern_i+1)*pattern_i/2):
                    random_index = np.random.randint(0, len(position_copypaste))
                    # if random_index not yet taken for a pattern
                    if position_copypaste[random_index] == 0:
                        if random_index > 0 and random_index < len(position_copypaste) -1 and \
                        position_copypaste[random_index - 1] != pattern_i and \
                        position_copypaste[random_index + 1] != pattern_i: ## index before and after the random index is not the same pattern
                            position_copypaste[random_index] = pattern_i
                        elif random_index == 0 and position_copypaste[random_index + 1] != pattern_i:
                            position_copypaste[random_index] = pattern_i
                        elif random_index == len(position_copypaste) - 1 and position_copypaste[random_index - 1] != pattern_i:
                            position_copypaste[random_index] = pattern_i
            return position_copypaste
        

    def copy_and_paste_jittered_pattern(self, times, indices, position_copypaste):
        patterns_info = {}
        for pattern_i in range(1, self.number_pattern + 1):
            startCPindex = np.where(position_copypaste == pattern_i)[0][0]
            start_idx = np.searchsorted(times, startCPindex * self.patternlength)
            end_idx = np.searchsorted(times, (startCPindex + 1) * self.patternlength)
            permuted_indices = np.random.permutation(self.number_neurons)
            length = int(self.number_neurons * self.portion_involved_pattern)
            afferents_in_pattern = permuted_indices[:length]
            afferents_not_in_pattern = permuted_indices[length:]

            time_window = times[start_idx: end_idx]
            indices_window = indices[start_idx: end_idx]
            tmp = np.isin(indices_window, afferents_in_pattern)
            time_window_pattern = time_window[tmp]
            indices_window_pattern = indices_window[tmp]

            time_window_pattern -= startCPindex * self.patternlength
            info_dic = {'time_window_pattern': time_window_pattern, 'indices_window_pattern': indices_window_pattern, 'afferents_in_pattern': afferents_in_pattern, 'afferents_not_in_pattern': afferents_not_in_pattern}
            patterns_info[f"pattern_{pattern_i}"] = info_dic

        times_final, indices_final = [], []
        for position_index, position_value in enumerate(position_copypaste):
            if position_value == 0:
                start_pattern = np.searchsorted(times, position_index * self.patternlength)
                end_pattern = np.searchsorted(times, (position_index + 1) * self.patternlength)
                times_final.append(times[start_pattern:end_pattern])
                indices_final.append(indices[start_pattern:end_pattern])
            else:
                info_dic = patterns_info[f"pattern_{position_value}"]
                time_window_pattern = info_dic['time_window_pattern']
                indices_window_pattern = info_dic['indices_window_pattern']
                afferents_in_pattern = info_dic['afferents_in_pattern']
                afferents_not_in_pattern = info_dic['afferents_not_in_pattern']
                
                timecopy = np.copy(time_window_pattern)
                indcopy = np.copy(indices_window_pattern)
                timecopy += position_index * self.patternlength
                timecopy = np.round(timecopy, 3)
                times_final.append(timecopy)
                indices_final.append(indcopy)

                start_idx = np.searchsorted(times, position_index * self.patternlength)
                end_idx = np.searchsorted(times, (position_index + 1) * self.patternlength)
                time_window = times[start_idx:end_idx]
                indices_window = indices[start_idx:end_idx]
                tmp = np.isin(indices_window, afferents_not_in_pattern)
                time_window_not_in_pattern = time_window[tmp]
                indices_window_not_in_pattern = indices_window[tmp]

                times_final.append(time_window_not_in_pattern)
                indices_final.append(indices_window_not_in_pattern)

                s1 = set(indcopy)
                s2 = set(indices_window_not_in_pattern)
                s3 = s1.isdisjoint(s2)
                if not s3:
                    print("Not disjoint")
                    print(s1)
                    print(s2)
                    print("\n\n")

        times_final = np.hstack(times_final)
        indices_final = np.hstack(indices_final)
        sortarray = times_final.argsort()
        indices_final = indices_final[sortarray]
        times_final = times_final[sortarray]
        return times_final, indices_final, patterns_info


    def triple_input_runtime(self, times, indices):
        # To shorten time spent on creating input, 150s input is tripled to give 450s
        times = np.concatenate((times, np.round(times + 150, 3), np.round(times + 300, 3)))
        indices = np.concatenate((indices, indices, indices))
        return times, indices
    
    def generate(self):
        times, indices = self.make_input()
        position_copypaste = self.make_pattern_presentation_array()
                                    
        times, indices, patterns_info = self.copy_and_paste_jittered_pattern(times, indices, position_copypaste)
        if self.tripling and self.runduration > 300:
            times, indices = self.triple_input_runtime(times, indices)
            position_copypaste = np.concatenate((position_copypaste, position_copypaste, position_copypaste))
        timing_pattern = np.where(position_copypaste > 0)[0] * self.patternlength

        indices = indices.astype(int)

        return times, indices, position_copypaste, patterns_info, timing_pattern