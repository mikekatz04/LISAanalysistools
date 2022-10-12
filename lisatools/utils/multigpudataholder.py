import cupy as xp
import numpy as np


class MultiGPUDataHolder:
    def __init__(self, gpus, channel1_data, channel2_data, channel1_psd, channel2_psd, df, base_injections=None, base_psd=None):

        if isinstance(gpus, int):
            gpus = [gpus]

        self.df = df

        if not isinstance(gpus, list) or not isinstance(gpus[0], int):
            raise ValueError("gpus must be an integer or a list of integers.")

        self.gpus = gpus
        self.num_gpus = len(gpus)
        # need to be numpy coming in to now make memory large
        self.ntemps, self.nwalkers, self.data_length = channel1_data.shape
        self.total_number = self.ntemps * self.nwalkers

        self.temp_indices = np.repeat(np.arange(self.ntemps), self.nwalkers).reshape(self.ntemps, self.nwalkers)
        self.walker_indices = np.tile(np.arange(self.nwalkers), (self.ntemps, 1)).flatten().reshape(self.ntemps, self.nwalkers)
        self.overall_indices_flat = np.arange(self.ntemps * self.nwalkers)
        self.overall_indices = self.overall_indices_flat.reshape(self.ntemps, self.nwalkers)

        self.base_injections = base_injections
        self.base_psd = base_psd

        self.map = self.overall_indices_flat.copy()

        num_per_split = self.total_number // self.num_gpus + 1 * (self.total_number % self.num_gpus != 0)
        # gpu arangement
        self.gpu_split_inds = np.arange(num_per_split, self.total_number, num_per_split)

        self.gpu_splits = np.split(self.overall_indices_flat, self.gpu_split_inds)
        self.gpus_for_each_data = [np.full_like(gpu_split, gpu) for gpu_split, gpu in zip( self.gpu_splits, self.gpus)]

        self.channel1_data = [_ for _ in range(self.num_gpus)]
        self.channel2_data = [_ for _ in range(self.num_gpus)]
        self.channel1_psd = [_ for _ in range(self.num_gpus)]
        self.channel2_psd = [_ for _ in range(self.num_gpus)]
        return_to_main = xp.cuda.runtime.getDevice()
        for gpu_i, (gpu, gpu_split) in enumerate(zip(self.gpus, self.gpu_splits)):
            temp_inds_gpu_here = self.temp_indices.flatten()[gpu_split]
            walker_inds_gpu_here = self.walker_indices.flatten()[gpu_split]

            with xp.cuda.device.Device(gpu):
                self.channel1_data[gpu_i] = xp.asarray(
                    channel1_data[(temp_inds_gpu_here, walker_inds_gpu_here)].flatten()
                )
                self.channel2_data[gpu_i] = xp.asarray(
                    channel2_data[(temp_inds_gpu_here, walker_inds_gpu_here)].flatten()
                )
                self.channel1_psd[gpu_i] = xp.asarray(
                    channel1_psd[(temp_inds_gpu_here, walker_inds_gpu_here)].flatten()
                )
                self.channel2_psd[gpu_i] = xp.asarray(
                    channel2_psd[(temp_inds_gpu_here, walker_inds_gpu_here)].flatten()
                )
        
        xp.cuda.runtime.setDevice(return_to_main)
        xp.cuda.runtime.deviceSynchronize()

    def reshape_list(self, input_value):
        return [
            self.reshape(tmp) for tmp in input_value
        ]

    def reshape(self, input_value):
        return input_value.reshape(-1, self.data_length)

    @property
    def data_list(self):
        return [self.channel1_data, self.channel2_data]

    @property
    def psd_list(self):
        return [self.channel1_psd, self.channel2_psd]

    @property
    def data_shaped(self):
        return [
            self.reshape_list(self.channel1_data),
            self.reshape_list(self.channel2_data),
        ]

    @property
    def psd_shaped(self):
        return [
            self.reshape_list(self.channel1_psd),
            self.reshape_list(self.channel2_psd),
        ]

    @property
    def map(self):
        return self._map

    @map.setter
    def map(self, map):
        if not isinstance(map, np.ndarray) or len(map) != self.total_number or map.dtype != np.int64:
            raise ValueError("map input must be a numpy array of np.int64 that is the same length as the number of gpu holder slots.")
        self._map = map

    @property
    def full_length(self):
        return self.ntemps * self.nwalkers * self.data_length

    def get_mapped_indices(self, inds_in):
        if (not isinstance(inds_in, np.ndarray) and not isinstance(inds_in, xp.ndarray)) or ((inds_in.dtype != np.int64 and inds_in.dtype != xp.int32)):
            raise ValueError("inds_in input must be a numpy array of np.int64.")

        if isinstance(inds_in, np.ndarray):
            xp_here = np
        else:
            xp_here = xp
        return xp_here.asarray(self.map)[inds_in]

    def get_inner_product(self, *args, map=True, use_cpu=False, **kwargs):

        return_to_main = xp.cuda.runtime.getDevice()

        inner_out = np.zeros((self.ntemps * self.nwalkers))

        inner_temp = [_ for _ in range(self.num_gpus)]
        
        if not use_cpu:
            for gpu_i, (gpu, gpu_split) in enumerate(zip(self.gpus, self.gpu_splits)):

                with xp.cuda.device.Device(gpu):
                    inner_here = self.df * 4 * xp.sum(
                        self.reshape(self.channel1_data[gpu_i].conj() * self.channel1_data[gpu_i] / self.channel1_psd[gpu_i])
                        + self.reshape(self.channel2_data[gpu_i].conj() * self.channel2_data[gpu_i] / self.channel2_psd[gpu_i]),
                        axis=1
                    ).get()
                    inner_temp[gpu_i] = inner_here

            for gpu_i, (gpu, gpu_split) in enumerate(zip(self.gpus, self.gpu_splits)):
                with xp.cuda.device.Device(gpu):
                    xp.cuda.runtime.deviceSynchronize()
        else:
            for gpu_i, (gpu, gpu_split) in enumerate(zip(self.gpus, self.gpu_splits)):
                inner_here = self.df * 4 * np.sum(
                        self.reshape(self.channel1_data[gpu_i].get().conj() * self.channel1_data[gpu_i].get() / self.channel1_psd[gpu_i].get())
                        + self.reshape(self.channel2_data[gpu_i].get().conj() * self.channel2_data[gpu_i].get() / self.channel2_psd[gpu_i].get()),
                        axis=1
                    )
                inner_temp[gpu_i] = inner_here

        xp.cuda.runtime.setDevice(return_to_main)
        xp.cuda.runtime.deviceSynchronize()
        inner_temp = np.concatenate(inner_temp)

        inner_out[self.map] = inner_temp.real

        return inner_out.reshape(self.ntemps, self.nwalkers)

    def get_ll(self, *args, **kwargs):
        inner_product = self.get_inner_product(*args, **kwargs)
        ll_out = -1/2 * inner_product
        return ll_out

    def multiply_data(self, val):
        return_to_main = xp.cuda.runtime.getDevice()
        if not isinstance(val, int) and not isinstance(val, float):
            raise NotImplementedError("val must be an int or float.")

        for gpu_i, (gpu, gpu_split) in enumerate(zip(self.gpus, self.gpu_splits)):
            with xp.cuda.device.Device(gpu): 
                for chan in range(len(self.data_list)):
                    self.data_list[chan][gpu_i] *= val

        for gpu_i, (gpu, gpu_split) in enumerate(zip(self.gpus, self.gpu_splits)):
            with xp.cuda.device.Device(gpu):
                xp.cuda.runtime.deviceSynchronize()

        xp.cuda.runtime.setDevice(return_to_main)
        xp.cuda.runtime.deviceSynchronize()

    def restore_base_injections(self):
        return_to_main = xp.cuda.runtime.getDevice()
        if self.base_injections is None or self.base_psd is None:
            raise ValueError("Must give base_injections and base_psd kwarg to __init__ to restore.")

        for gpu_i, (gpu, gpu_split) in enumerate(zip(self.gpus, self.gpu_splits)):
            with xp.cuda.device.Device(gpu): 
                for chan in range(len(self.data_list)):
                    tmp = self.data_list[chan][gpu_i].reshape(self.ntemps * self.nwalkers, -1)
                    tmp[:] = xp.asarray(self.base_injections[chan])[None, :]
                    self.data_list[chan][gpu_i] = tmp.flatten()

                    tmp = self.psd_list[chan][gpu_i].reshape(self.ntemps * self.nwalkers, -1)
                    tmp[:] = xp.asarray(self.base_psd[chan])[None, :]
                    self.psd_list[chan][gpu_i] = tmp.flatten()

        for gpu_i, (gpu, gpu_split) in enumerate(zip(self.gpus, self.gpu_splits)):
            with xp.cuda.device.Device(gpu):
                xp.cuda.runtime.deviceSynchronize()

        xp.cuda.runtime.setDevice(return_to_main)
        xp.cuda.runtime.deviceSynchronize()

    def get_injection_inner_product(self, *args, **kwargs):

        inner_out = self.df * 4 * np.sum(
            self.base_injections[0].conj() * self.base_injections[0] / self.base_psd[0]
            + self.base_injections[1].conj() * self.base_injections[1] / self.base_psd[1],
        )
        return inner_out




if __name__ == "__main__":
    ntemps = 2
    nwalkers = 100
    data_length = int(1.6e5)
    nchannels = 2
    df = 3e-8

    data_A = np.ones((ntemps, nwalkers, data_length), dtype=complex)
    data_E = np.ones((ntemps, nwalkers, data_length), dtype=complex)

    psd_A = np.ones((ntemps, nwalkers, data_length), dtype=complex)
    psd_E = np.ones((ntemps, nwalkers, data_length), dtype=complex)

    gpus = [5, 6]
    mg = MultiGPUDataHolder(gpus, data_A, data_E, psd_A, psd_E, df)

    check1 = mg.get_mapped_indices(np.arange(len(mg.overall_indices_flat)))

    mg.map = np.random.choice(mg.overall_indices_flat, len(mg.overall_indices_flat), replace=False) 

    check2 = mg.get_mapped_indices(np.arange(len(mg.overall_indices_flat)))

    check3 = mg.get_ll()
    breakpoint()
    

    