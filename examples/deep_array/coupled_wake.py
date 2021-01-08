import scipy.optimize as opt
import numpy as np

# couple floris and top down model together to update wake expansion parmeters
class CoupledModel():

    def __init__(self, zh, z0, D, U_infty, H, N):

        # initialize the top down model
        self.zh = zh  # hub heigh
        self.z0 = z0  # roughness factor
        self.D = D    # average turbine diameter
        self.H = H    # typically 750m
        self.N = N    # number of turbines

        self.R = self.D * 0.5  # radius
        self.kappa = 0.4       # constant from literature
        self.z0_lo = self.z0   # XXXX
        self.Uinfty = U_infty  # freestream velocity
        self.ustar = self.Uinfty * self.kappa / np.log(self.zh / self.z0)  # friction velocity

    def calc_model(self, fi, td, Ctp, Cpp, alpha, prefactor):

        # farm thrust coefficient
        cft_prev = np.ones(fi.N) * 100

        # wake expansion for each turbine
        kw = fi.ustar / fi.Uinfty * np.ones(fi.N)

        # calculate the wake model with the new wake expansion
        fi.calc(Ctp, Cpp, kw)
        cft = wm.cft

        # hub height velocity?
        uh = np.zeros(wm.N)
        while np.sum((cft - cft_prev) ** 2) > 0.00001:
            cft_prev = np.array(cft)
            nu_w_star = 28 * np.sqrt(0.5 * cft)

            for i in range(wm.N):
                # compute the top down model with the new cft, nu_w_star, alpha
                td[i].calc(cft[i], nu_w_star[i], alpha, np.abs(wm.s[i, 0] - wm.xo[i]))

                # wake expansion from the top down model
                kw[i] = td[i].kw

                # uh from the top down model
                uh[i] = td[i].uh

            # compute the wake model with new wake expansion parameters
            wm.calc(Ctp, Cpp, kw)

            # XXX
            cft = np.array(wm.cft)

        # # power from the turbine - check to make sure it is reasonable
        # Phat = ((np.sum(wm.P)) * prefactor) / (1000000)

        return kw, uh

    def fmin_fun(self,p, wm, td, Ctp, Cpp, prefactor):

        # wake expansion parameter
        alpha = p[0]

        # difference between top down and wake model
        kw, uh = calc_model(wm, td, Ctp, Cpp, alpha, prefactor)

        print(alpha, np.sum((wm.uh - uh) ** 2))
        return np.sum((wm.uh - uh) ** 2)

    #     print(alpha, np.sum((wm.uh[ind] - uh[ind])**2))
    #     return np.sum((wm.uh[ind] - uh[ind])**2)

    def combine(self,Ctp, Cpp, D, s, z0, zh, H, ui, yi, N, prefactor):

        # Initialize wake model
        wm = WakeModel(D, s, z0, zh, ui, yi, N)  # this is where FLORIS goes

        # initialize top down model for each turbine
        td = [None] * wm.N
        for i in range(wm.N):
            td[i] = TopDown(wm.zh, wm.z0, wm.D[i], wm.ustari[i], H)

        #
        o = opt.minimize(fmin_fun, [2.0], args=(wm, td, Ctp, Cpp, prefactor), tol=0.0001, bounds=opt.Bounds(0, 100))
        alpha = o.x[0]
        print('alpha, end', alpha)
        #     alpha = 0.5 * o.x[0]
        calc_model(wm, td, Ctp, Cpp, alpha, prefactor)
        Phat = ((np.sum(wm.P)) * prefactor) / (1000000)
        print('phat_end', Phat)
        print('kw end', wm.kw)
        return wm, td, alpha