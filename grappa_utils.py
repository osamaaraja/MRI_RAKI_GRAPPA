import numpy as np

from numpy.fft import fftshift, ifftshift, ifft2, fft2
from scipy.linalg import fractional_matrix_power
import matplotlib.pyplot as plt

def fft2c(x, axes=(-2, -1)):
    return (1 / np.sqrt(np.size(x))) * fftshift(fft2(ifftshift(x, axes=axes), axes=axes), axes=axes)

def ifft2c(x, axes=(-2, -1)):
    return np.sqrt(np.size(x)) * fftshift(ifft2(ifftshift(x, axes=axes), axes=axes), axes=axes)

class GRAPPA:
    def __init__(self, kspace, nACS=24, kernel_size=(2, 3)):

        self.kspace = kspace
        self.nACS = nACS
        self.PE, self.RO, self.nCoil = kspace.shape
        self.kernel_PE, self.kernel_RO = kernel_size

    def _set_params(self, R):

        self.R = R
        self.zf_kspace = self._undersample()
        self.acs = self._get_acs()

        self.block_w = self.kernel_RO
        self.block_h = self.R * (self.kernel_PE - 1) + 1

        self.nb = (self.nACS - self.block_h + 1) * (self.RO - self.block_w + 1)
        self.nkc = self.kernel_RO * self.kernel_PE * self.nCoil

    def _get_acs(self):

        acs_u = self.PE // 2 - self.nACS // 2
        acs_d = self.PE // 2 + self.nACS // 2

        return self.kspace[acs_u:acs_d]

    def _undersample(self):

        mask = self._get_mask()
        return self.kspace * mask

    def _get_mask(self):

        mask = np.zeros_like(self.kspace)
        mask[::self.R] = 1   # start:stop:step
        return mask

    def _extract(self):

        src = np.zeros((self.nb, self.nkc), dtype=self.kspace.dtype)
        target = np.zeros((self.R - 1, self.nb, self.nCoil), dtype=self.kspace.dtype)

        box_idx = 0
        for idx_RO in range(self.kernel_RO // 2, self.RO - self.kernel_RO // 2):
            for idx_ACS in range(self.nACS - self.block_h + 1):
                src[box_idx] = np.array(
                        [
                            [
                                self.acs[idx_ACS + dy * self.R, idx_RO + dx] for dy in range(self.kernel_PE)
                            ]
                            for dx in range(-(self.kernel_RO // 2), self.kernel_RO // 2 + 1)
                        ]
                ).flatten()

                for dy in range(self.R - 1):
                    target[dy, box_idx] = np.array(
                            self.acs[idx_ACS + (self.kernel_PE // 2 - 1) * self.R + dy + 1, idx_RO]).flatten()
                box_idx += 1

        return src, target

    def _interpolation(self, zp_kspace):

        interpolated = zp_kspace.copy()
        for idx_RO in range(self.kernel_RO // 2, self.RO - self.kernel_RO // 2):
            for idx_PE in range(0, self.PE - self.block_h + 1, self.R):
                source = np.array(
                        [[zp_kspace[idx_PE + self.R * dy, idx_RO + dx] for dy in range(self.kernel_PE)]
                         for dx in range(-(self.kernel_RO // 2), self.kernel_RO // 2 + 1)]
                ).flatten()

                for dy in range(self.R - 1):
                    interpolated[idx_PE + (self.kernel_PE // 2 - 1) * self.R + dy + 1, idx_RO] = np.dot(source,
                                                                                                        self.ws[dy])
        return interpolated

    def _zero_padding(self):

        zp_kspace = np.zeros((self.PE + self.R * 2, self.RO + self.kernel_RO // 2 * 2, self.nCoil),
                             dtype=self.kspace.dtype)
        zp_kspace[self.R:self.PE + self.R, self.kernel_RO // 2: self.RO + self.kernel_RO // 2] = self.zf_kspace

        return zp_kspace[self.R:self.PE + self.R, self.kernel_RO // 2: self.RO + self.kernel_RO // 2]

    '''
    def _crop2original(self, interpolated):

        return interpolated
    '''

    def grappa(self, R, flag_acs=False):

        self._set_params(R)
        src, targ = self._extract() # extraction of source and target

        # todo
        self.ws = np.linalg.pinv(src)[None, ...] @ targ # the grappa weights

        zp_kdata = self._zero_padding()
        #print(f"the size of the zero padded data is {zp_kdata.shape}")

        interpolated = self._interpolation(zp_kdata)
        #interpolated = self._crop2original(interpolated)

        '''
        if flag_acs:
            # todo
            pass
        '''

        return interpolated