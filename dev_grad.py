import numpy as np
import matplotlib.pyplot as plt

from lisatools.detector import EqualArmlengthOrbits
from lisatools.utils.constants import *

class GBWaveform:
    def __init__(self, orbits):
        self.orbits = orbits

    def add_Psi_piece(self, t, params, sign, rec, em, delays=[]):
        link = int(str(rec + 1) + str(em + 1))
        n_ij = self.orbits.get_normal_unit_vec(t, link)
        r_j = self.orbits.get_pos(t, rec + 1)
        L_ij = np.zeros_like(t)
        last_spacecraft = rec
        for delay in delays:
            multiplier, next_spacecraft = delay
            delay_link = int(str(last_spacecraft + 1) + str(next_spacecraft + 1))
            L_ij += self.orbits.get_light_travel_times(t, delay_link)
            last_spacecraft = next_spacecraft
        output = self.Psi_of_t(t, params, sign, n_ij, r_j, L_ij)
        return output
    
    def grad_add_Psi_piece(self, t, params, sign, rec, em, delays=[]):
        link = int(str(rec + 1) + str(em + 1))
        n_ij = self.orbits.get_normal_unit_vec(t, link)
        r_j = self.orbits.get_pos(t, rec + 1)
        L_ij = np.zeros_like(t)
        last_spacecraft = rec
        for delay in delays:
            multiplier, next_spacecraft = delay
            delay_link = int(str(last_spacecraft + 1) + str(next_spacecraft + 1))
            L_ij += self.orbits.get_light_travel_times(t, delay_link)
            last_spacecraft = next_spacecraft
        output = self.grad_Psi_of_t(t, params, sign, n_ij, r_j, L_ij)
        return output
    
    def grad_X_of_t(self, t, params):
        output = np.zeros((t.shape[0], len(params)), dtype=complex)
        rec = 0
        em = 1
        delays = [(2, 1)]
        output += self.grad_add_Psi_piece(t, params, -1, rec, em, delays)
        return output
    
    def X_of_t(self, t, params):
        
        output = np.zeros_like(t, dtype=complex)
        rec = 0
        em = 1
        delays = [(2, 1)]
        output += self.add_Psi_piece(t, params, -1, rec, em, delays)
        
        rec = 2
        em = 1
        delays = [(1, 1)]
        output -= self.add_Psi_piece(t, params, -1, rec, em, delays)
        
        rec = 0
        em = 2
        delays = [(2, 2)]
        output -= self.add_Psi_piece(t, params, +1, rec, em, delays)

        rec = 1
        em = 2
        delays = [(1, 2)]
        output += self.add_Psi_piece(t, params, +1, rec, em, delays)
        
        rec = 2
        em = 1
        delays = [(1, 1)]
        output += self.add_Psi_piece(t, params, +1, rec, em, delays)
        
        rec = 0
        em = 1
        delays = []
        output -= self.add_Psi_piece(t, params, +1, rec, em, delays)
        
        rec = 1
        em = 2
        delays = [(1, 2)]
        output -= self.add_Psi_piece(t, params, -1, rec, em, delays)
        
        rec = 0
        em = 2
        delays = []
        output += self.add_Psi_piece(t, params, -1, rec, em, delays)
        
        rec = 0
        em = 2
        delays = [(2, 2), (2, 1)]
        output += self.add_Psi_piece(t, params, +1, rec, em, delays)
        
        rec = 1
        em = 2
        delays = [(1, 2), (2, 1)]
        output -= self.add_Psi_piece(t, params, +1, rec, em, delays)
        
        rec = 0
        em = 1
        delays = [(2, 1), (2, 2)]
        output -= self.add_Psi_piece(t, params, -1, rec, em, delays)
        
        rec = 2
        em = 1
        delays = [(1, 1), (2, 2)]
        output += self.add_Psi_piece(t, params, -1, rec, em, delays)
        
        rec = 1
        em = 2
        delays = [(1, 2), (2, 1)]
        output += self.add_Psi_piece(t, params, -1, rec, em, delays)
        
        rec = 0
        em = 2
        delays = [(2, 1)]
        output -= self.add_Psi_piece(t, params, -1, rec, em, delays)
        
        rec = 2
        em = 1
        delays = [(1, 1), (2, 2)]
        output -= self.add_Psi_piece(t, params, +1, rec, em, delays)
        
        rec = 0
        em = 1
        delays = [(2, 2)]
        output += self.add_Psi_piece(t, params, +1, rec, em, delays)
        
        return output
    
    def Psi_of_t(self, t, params, sign, n_ij, r_j, L_ij):
        lam = self.get_lam(params)
        beta = self.get_beta(params)
        k_hat = np.array([np.cos(beta) * np.cos(lam), np.cos(beta) * np.sin(lam), -np.sin(beta)])
    
        e_plus = self.e_plus(params)
        e_cross = self.e_cross(params)

        h_plus_re = self.h_plus_re(t, params, r_j, L_ij)
        h_cross_re = self.h_cross_re(t, params, r_j, L_ij)

        h_plus_im = self.h_plus_im(t, params, r_j, L_ij)
        h_cross_im = self.h_cross_im(t, params, r_j, L_ij)
        
        d_plus = (np.einsum("...i,...i->...", n_ij, np.einsum("ij,...j->...i", e_plus, n_ij))) / (2 * (1. + sign * np.einsum("i,...i->...", k_hat, n_ij)))
        d_cross = (np.einsum("...i,...i->...", n_ij, np.einsum("ij,...j->...i", e_cross, n_ij))) / (2 * (1. + sign * np.einsum("i,...i->...", k_hat, n_ij)))
        
        h_re = d_plus * h_plus_re + d_cross * h_cross_re
        h_im = d_plus * h_plus_im + d_cross * h_cross_im
        return h_re + 1j * h_im
    
    def k_hat(self, params):
        lam = self.get_lam(params)
        beta = self.get_beta(params)
        k_hat = np.array([np.cos(beta) * np.cos(lam), np.cos(beta) * np.sin(lam), -np.sin(beta)])
        return k_hat
    
    def grad_k_hat(self, params):
        lam = self.get_lam(params)
        beta = self.get_beta(params)
        grad_k_hat = np.zeros((len(params), 3))
        grad_k_hat[6] = np.array([-np.cos(beta) * np.sin(lam), np.cos(beta) * np.cos(lam), 0.0])
        grad_k_hat[7] = np.array([-np.sin(beta) * np.cos(lam), -np.sin(beta) * np.sin(lam), -np.cos(beta)])
        
        return grad_k_hat

    def grad_Psi_of_t(self, t, params, sign, n_ij, r_j, L_ij):
        
        k_hat = self.k_hat(params)

        e_plus = self.e_plus(params)
        e_cross = self.e_cross(params)

        h_plus_re = self.h_plus_re(t, params, r_j, L_ij)
        h_cross_re = self.h_cross_re(t, params, r_j, L_ij)

        h_plus_im = self.h_plus_im(t, params, r_j, L_ij)
        h_cross_im = self.h_cross_im(t, params, r_j, L_ij)
        
        d_plus = (np.einsum("...i,...i->...", n_ij, np.einsum("ij,...j->...i", e_plus, n_ij))) / (2 * (1. + sign * np.einsum("i,...i->...", k_hat, n_ij)))
        d_cross = (np.einsum("...i,...i->...", n_ij, np.einsum("ij,...j->...i", e_cross, n_ij))) / (2 * (1. + sign * np.einsum("i,...i->...", k_hat, n_ij)))
        
        grad_k_hat = self.grad_k_hat(params)
    
        grad_e_plus = self.grad_e_plus(params)
        grad_e_cross = self.grad_e_cross(params)

        grad_h_plus_re = self.grad_h_plus_re(t, params, r_j, L_ij)
        grad_h_cross_re = self.grad_h_cross_re(t, params, r_j, L_ij)

        grad_h_plus_im = self.grad_h_plus_im(t, params, r_j, L_ij)
        grad_h_cross_im = self.grad_h_cross_im(t, params, r_j, L_ij)
        
        grad_d_plus = (np.einsum("...j,...ij->...i", n_ij, np.einsum("ijk,...k->...ij", grad_e_plus, n_ij))) / (2 * (1. + sign * np.einsum("ij,...i->...j", grad_k_hat.T, n_ij)))
        grad_d_cross = (np.einsum("...j,...ij->...i", n_ij, np.einsum("ijk,...k->...ij", grad_e_cross, n_ij))) / (2 * (1. + sign * np.einsum("ij,...i->...j", grad_k_hat.T, n_ij)))
        
        grad_h_re = (
            grad_d_plus * h_plus_re[:, None]  
            + d_plus[:, None] * grad_h_plus_re 
            + grad_d_cross * h_cross_re[:, None]  
            + d_cross[:, None] * grad_h_cross_re
        )

        grad_h_im = (
            grad_d_plus * h_plus_im[:, None]  
            + d_plus[:, None] * grad_h_plus_im 
            + grad_d_cross * h_cross_im[:, None]  
            + d_cross[:, None] * grad_h_cross_im
        )

        return grad_h_re + 1j * grad_h_im

    def A_plus(self, params, trig):
        A = params[0]
        iota = params[4]
        psi = params[5]
        if trig == "cos":
            return +A * (1 + np.cos(iota) ** 2) / 2. * np.cos(2 * psi)
        elif trig == "sin":
            return -A * np.cos(iota) * np.sin(2 * psi)
        else:
            raise ValueError
        
    @property
    def A_index(self):
        return 0
    
    def get_A(self, params):
        return params[self.A_index]
    
    @property
    def f0_index(self):
        return 1
    
    def get_f0(self, params):
        return params[self.f0_index]
    
    @property
    def fdot0_index(self):
        return 2
    
    def get_fdot0(self, params):
        return params[self.fdot0_index]
    
    @property
    def phi0_index(self):
        return 3
    
    def get_phi0(self, params):
        return params[self.phi0_index]
    
    @property
    def iota_index(self):
        return 4
    
    def get_iota(self, params):
        return params[self.iota_index]
    
    @property
    def psi_index(self):
        return 5
    
    def get_psi(self, params):
        return params[self.psi_index]
        
    def grad_A_plus(self, params, trig):
        A = self.get_A(params)
        iota = self.get_iota(params)
        psi = self.get_psi(params)

        grad_A_plus = np.zeros(len(params))

        if trig == "cos":
            grad_A_plus[self.A_index] = (1 + np.cos(iota) ** 2) / 2. * np.cos(2 * psi)
            grad_A_plus[self.iota_index] = -A * np.cos(iota) * np.sin(iota) * np.cos(2 * psi)
            grad_A_plus[self.psi_index] = -2 * A * (1 + np.cos(iota) ** 2) / 2. * np.sin(2 * psi)
            
        elif trig == "sin":
            grad_A_plus[self.A_index] = -np.cos(iota) * np.sin(2 * psi)
            grad_A_plus[self.iota_index] = +A * np.sin(iota) * np.sin(2 * psi)
            grad_A_plus[self.psi_index] = -2 * A * np.cos(iota) * np.cos(2 * psi)
            
        else:
            raise ValueError
        
        return grad_A_plus
        
    def grad_A_cross(self, params, trig):
        A = self.get_A(params)
        iota = self.get_iota(params)
        psi = self.get_psi(params)
        
        grad_A_cross = np.zeros(len(params))

        if trig == "cos":
            grad_A_cross[self.A_index] = -(1 + np.cos(iota) ** 2) / 2. * np.sin(2 * psi)
            grad_A_cross[self.iota_index] = +A * np.cos(iota) * np.sin(iota) * np.sin(2 * psi)
            grad_A_cross[self.psi_index] = -2 * A * (1 + np.cos(iota) ** 2) / 2. * np.cos(2 * psi)
            
        elif trig == "sin":
            grad_A_cross[self.A_index] = -np.cos(iota) * np.cos(2 * psi)
            grad_A_cross[self.iota_index] = +A * np.sin(iota) * np.cos(2 * psi)
            grad_A_cross[self.psi_index] = +2 * A * np.cos(iota) * np.sin(2 * psi)
            
        else:
            raise ValueError
        return grad_A_cross
        
    def A_cross(self, params, trig):
        A = params[0]
        iota = params[4]
        psi = params[5]
        if trig == "cos":
            return -A * (1 + np.cos(iota) ** 2) / 2. * np.sin(2 * psi)
        elif trig == "sin":
            return -A * np.cos(iota) * np.cos(2 * psi)
        else:
            raise ValueError
        
    def Phi_of_t(self, t, params, r_j, L_ij):
        xi = self.xi_of_t(t, params, r_j, L_ij)

        f0 = self.get_f0(params)
        fdot0 = self.get_fdot0(params)
        phi0 = self.get_phi0(params)
        output = phi0 + 2 * np.pi * (f0 * xi + 1./2. * fdot0 * xi ** 2)
        return output
    
    def grad_Phi_of_t(self, t, params, r_j, L_ij):

        xi = self.xi_of_t(t, params, r_j, L_ij)
        f0 = self.get_f0(params)
        fdot0 = self.get_fdot0(params)
        phi0 = self.get_phi0(params)

        grad_Phi_of_t = np.zeros((t.shape[0], len(params)))
        grad_Phi_of_t[:, self.phi0_index] = 1.0
        grad_Phi_of_t[:, self.f0_index] = 2 * np.pi * xi
        grad_Phi_of_t[:, self.fdot0_index] = np.pi * xi ** 2

        grad_Phi_of_t[:, self.lam_index] = (
            2 * np.pi * (f0 + fdot0 * xi) 
        )
        grad_Phi_of_t[:, self.beta_index] = (
            2 * np.pi * (f0 + fdot0 * xi) 
        )
        return grad_Phi_of_t
    
    def xi_of_t(self, t, params, r_j, L_ij):
        k_hat = self.k_hat(params)
        xi = t - np.dot(k_hat, r_j.T) - L_ij
        return xi 
    
    def grad_xi_of_t(self, t, params, r_j, L_ij):
        grad_k_hat = self.grad_k_hat(params)
        grad_xi = -np.dot(r_j, grad_k_hat.T)
        return grad_xi 
    
    def h_plus_re(self, t, params, r_j, L_ij):
        output = (
            self.A_plus(params, "cos") * np.cos(self.Phi_of_t(t, params, r_j, L_ij))
            + self.A_plus(params, "sin") * np.sin(self.Phi_of_t(t, params, r_j, L_ij))
        )
        return output
    
    def grad_h_plus_re(self, t, params, r_j, L_ij):
        A_plus_cos = self.A_plus(params, "cos")
        A_plus_sin = self.A_plus(params, "sin")

        grad_A_plus_cos = self.grad_A_plus(params, "cos")
        grad_A_plus_sin = self.grad_A_plus(params, "sin")
        
        Phi_of_t = self.Phi_of_t(t, params, r_j, L_ij)
        
        grad_Phi_of_t = self.grad_Phi_of_t(t, params, r_j, L_ij)

        inds_sky = np.array([self.lam_index, self.beta_index])

        # the only parameters that relate to xi is beta and lam through k_hat
        grad_Phi_of_t[:, inds_sky] *= self.grad_xi_of_t(t, params, r_j, L_ij)[:, inds_sky]
        
        output = (
            grad_A_plus_cos[None, :] * np.cos(Phi_of_t)[:, None]
            - A_plus_cos * np.sin(Phi_of_t)[:, None] * grad_Phi_of_t
            
            + grad_A_plus_sin[None, :] * np.sin(Phi_of_t)[:, None]
            + A_plus_sin * np.cos(Phi_of_t)[:, None] * grad_Phi_of_t
        )
        return output
    
    def h_plus_im(self, t, params, r_j, L_ij):
        output = (
            self.A_plus(params, "cos") * np.sin(self.Phi_of_t(t, params, r_j, L_ij))
            - self.A_plus(params, "sin") * np.cos(self.Phi_of_t(t, params, r_j, L_ij))
        )
        return output
    
    def grad_h_plus_im(self, t, params, r_j, L_ij):

        A_plus_cos = self.A_plus(params, "cos")
        A_plus_sin = self.A_plus(params, "sin")

        grad_A_plus_cos = self.grad_A_plus(params, "cos")
        grad_A_plus_sin = self.grad_A_plus(params, "sin")
        
        Phi_of_t = self.Phi_of_t(t, params, r_j, L_ij)
        
        grad_Phi_of_t = self.grad_Phi_of_t(t, params, r_j, L_ij)

        inds_sky = np.array([self.lam_index, self.beta_index])

        grad_Phi_of_t[:, inds_sky] *= self.grad_xi_of_t(t, params, r_j, L_ij)[:, inds_sky]
        
        output = (
            grad_A_plus_cos[None, :] * np.sin(Phi_of_t)[:, None]
            + A_plus_cos * np.cos(Phi_of_t)[:, None] * grad_Phi_of_t
            - grad_A_plus_sin[None, :] * np.cos(Phi_of_t)[:, None]
            + A_plus_sin * np.sin(Phi_of_t)[:, None] * grad_Phi_of_t
        )
        return output
    
    def h_cross_re(self, t, params, r_j, L_ij):
        output = (
            self.A_cross(params, "cos") * np.cos(self.Phi_of_t(t, params, r_j, L_ij))
            + self.A_cross(params, "sin") * np.sin(self.Phi_of_t(t, params, r_j, L_ij))
        )
        return output
    
    def grad_h_cross_re(self, t, params, r_j, L_ij):
        A_cross_cos = self.A_cross(params, "cos")
        A_cross_sin = self.A_cross(params, "sin")

        grad_A_cross_cos = self.grad_A_cross(params, "cos")
        grad_A_cross_sin = self.grad_A_cross(params, "sin")
        
        Phi_of_t = self.Phi_of_t(t, params, r_j, L_ij)
        
        grad_Phi_of_t = self.grad_Phi_of_t(t, params, r_j, L_ij)

        inds_sky = np.array([self.lam_index, self.beta_index])

        grad_Phi_of_t[:, inds_sky] *= self.grad_xi_of_t(t, params, r_j, L_ij)[:, inds_sky]
        
        output = (
            grad_A_cross_cos[None, :] * np.cos(Phi_of_t)[:, None]
            - A_cross_cos * np.sin(Phi_of_t)[:, None] * grad_Phi_of_t
            + grad_A_cross_sin[None, :] * np.sin(Phi_of_t)[:, None]
            + A_cross_sin * np.cos(Phi_of_t)[:, None] * grad_Phi_of_t
        
        )
        return output
    
    def h_cross_im(self, t, params, r_j, L_ij):
        output = (
            self.A_cross(params, "cos") * np.sin(self.Phi_of_t(t, params, r_j, L_ij))
            - self.A_cross(params, "sin") * np.cos(self.Phi_of_t(t, params, r_j, L_ij))
        )
        return output
    
    def grad_h_cross_im(self, t, params, r_j, L_ij):

        A_cross_cos = self.A_cross(params, "cos")
        A_cross_sin = self.A_cross(params, "sin")

        grad_A_cross_cos = self.grad_A_cross(params, "cos")
        grad_A_cross_sin = self.grad_A_cross(params, "sin")
        
        Phi_of_t = self.Phi_of_t(t, params, r_j, L_ij)
        
        grad_Phi_of_t = self.grad_Phi_of_t(t, params, r_j, L_ij)

        inds_sky = np.array([self.lam_index, self.beta_index])

        grad_Phi_of_t[:, inds_sky] *= self.grad_xi_of_t(t, params, r_j, L_ij)[:, inds_sky]
        
        output = (
            grad_A_cross_cos[None, :] * np.sin(Phi_of_t)[:, None]
            + A_cross_cos * np.cos(Phi_of_t)[:, None] * grad_Phi_of_t
            - grad_A_cross_sin[None, :] * np.cos(Phi_of_t)[:, None]
            + A_cross_sin * np.sin(Phi_of_t)[:, None] * grad_Phi_of_t
        )
        return output
    
    def e_plus(self, params):
        u_hat = self.u_hat(params)
        v_hat = self.v_hat(params)
        output = np.outer(v_hat, v_hat) - np.outer(u_hat, u_hat)
        return output
    
    def grad_e_plus(self, params):
        grad_u_hat = self.grad_u_hat(params)
        grad_v_hat = self.grad_v_hat(params)
        output = np.einsum("...i,...j->...ij", grad_v_hat, grad_v_hat) - np.einsum("...i,...j->...ij", grad_u_hat, grad_u_hat)
        return output
    
    def v_hat(self, params):
        lam = self.get_lam(params)
        beta = self.get_beta(params)
        return np.array([-np.sin(beta) * np.cos(lam), -np.sin(beta) * np.sin(lam), np.cos(beta)])
    
    def grad_v_hat(self, params):
        lam = self.get_lam(params)
        beta = self.get_beta(params)
        grad_v_hat = np.zeros((len(params), 3))
        grad_v_hat[self.lam_index] = np.array([+np.sin(beta) * np.sin(lam), -np.sin(beta) * np.cos(lam), 0.0])
        grad_v_hat[self.beta_index] = np.array([-np.cos(beta) * np.cos(lam), -np.cos(beta) * np.sin(lam), -np.sin(beta)])
        return grad_v_hat
    
    def u_hat(self, params):
        lam = self.get_lam(params)
        beta = self.get_beta(params)
        return np.array([np.sin(lam), np.cos(lam), 0.])
    
    def grad_u_hat(self, params):
        lam = self.get_lam(params)
        beta = self.get_beta(params)
        grad_u_hat = np.zeros((len(params), 3))
        grad_u_hat[self.lam_index] = np.array([np.cos(lam), -np.sin(lam), 0.])
        grad_u_hat[self.beta_index] = np.array([0.0, 0.0, 0.])
        return grad_u_hat
    
    @property
    def lam_index(self):
        return 6
    
    def get_lam(self, params):
        return params[self.lam_index]
    
    @property
    def beta_index(self):
        return 7
    
    def get_beta(self, params):
        return params[self.beta_index]
    
    def e_cross(self, params):
        u_hat = self.u_hat(params)
        v_hat = self.v_hat(params)
        output = np.outer(v_hat, u_hat) + np.outer(u_hat, v_hat)
        return output
    
    def grad_e_cross(self, params):
        grad_u_hat = self.grad_u_hat(params)
        grad_v_hat = self.grad_v_hat(params)
        output = np.einsum("...i,...j->...ij", grad_v_hat, grad_u_hat) + np.einsum("...i,...j->...ij", grad_u_hat, grad_v_hat)
        return output

    def n_vec(self, t, i, j):
        link = int(str(i + 1) + str(j + 1))
        return orbits.get_normal_unit_vec(t, link)


if __name__ == "__main__":
    orbits = EqualArmlengthOrbits()
    orbits.configure(linear_interp_setup=True)

    dt = 10000.0
    Tobs = (Nobs := int(YRSID_SI / dt)) * dt
    t = np.arange(Nobs) * dt
    params = [1e-22, 4e-3, 1e-16, np.pi / 3., np.pi/5., np.pi / 7., 5.0239482, -0.4]
    gb = GBWaveform(orbits)
    out = gb.X_of_t(t, params)
    grad_out = gb.grad_X_of_t(t, params)
    breakpoint()