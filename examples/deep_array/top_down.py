import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import copy
import shapely.geometry as geom
import scipy.spatial as spt
import scipy.interpolate as interp

class TopDown:

    def __init__(self, zh, z0, D, U_infty, H, N, layout_x, layout_y, fi):

        # initialize the top down model
        self.zh = zh  # hub heigh
        self.z0 = z0  # roughness factor
        self.D = D    # average turbine diameter
        self.H = H    # typically 750m
        self.N = N    # number of turbines

        self.R = self.D * 0.5  # radius
        self.kappa = 0.4       # constant from literature
        self.z0_lo = self.z0   # surface roughness
        self.Uinfty = U_infty  # freestream velocity
        self.ustar = self.Uinfty * self.kappa / np.log(self.zh / self.z0)  # friction velocity

        # floris wake model
        self.fi = fi

        # locations
        self.layout_x = layout_x
        self.layout_y = layout_y
        locsx = np.reshape(layout_x, (self.N, 1))
        locsy = np.reshape(layout_y, (self.N, 1))
        self.s = np.concatenate((locsx, locsy), axis=1)

        # compute the start of the boundary layer
        self.start_boundary_layer()

    def start_boundary_layer(self):

        # Determine initial trip line for IBL development based on convex hull

        D = self.D * np.ones(self.N)

        hull = spt.ConvexHull(self.s)
        outside_N = len(hull.vertices)
        new_v = np.zeros((outside_N, 2))
        self.center = np.zeros((outside_N, 2))
        self.center[:, 0] = (max(self.s[hull.vertices, 0]) - min(self.s[hull.vertices, 0])) / 2 + min(
            self.s[hull.vertices, 0])
        self.center[:, 1] = (max(self.s[hull.vertices, 1]) - min(self.s[hull.vertices, 1])) / 2 + min(
            self.s[hull.vertices, 1])
        xdis = self.center[:, 0] - self.s[hull.vertices, 0]
        ydis = self.center[:, 1] - self.s[hull.vertices, 1]

        L = np.sqrt(ydis ** 2 + xdis ** 2)
        aprime = (D[0:outside_N] / L) * ydis
        bprime = (D[0:outside_N] / L) * xdis
        new_v[:, 0] = self.s[hull.vertices, 0] - bprime
        new_v[:, 1] = self.s[hull.vertices, 1] - aprime
        front_half = new_v[new_v[:, 0] < self.center[:, 0]]
        f2 = interp.interp1d(front_half[:, 1], front_half[:, 0], kind='linear', fill_value='extrapolate')

        self.xo = np.zeros(self.N)
        self.xo = f2(self.s[:, 1])

    def calc(self, cft, x):

        # the planform thrust coefficient that represents the momentum extracted from the flow by the turbines’ axial
        # flow resistance
        self.cft = cft
        nu_w_star = 28 * np.sqrt(0.5 * cft)
        self.x = x - self.xo  # downstream distances

        # model parameters
        beta = nu_w_star / ( 1 + nu_w_star)

        # the roughness height due to the wind farm
        self.z0_hi = self.zh * ( 1 + self.R / self.zh ) ** beta \
                   * np.exp( -( 0.5 * self.cft / self.kappa**2 + ( np.log( self.zh/self.z0_lo * ( 1-self.R/self.zh ) ** beta)) ** -2) ** -0.5)

        # boundary layer height
        delta = self.zh + self.z0_hi * ( self.x/self.z0_hi ) ** (4/5) # + self.R
        delta = np.minimum(delta, self.H)

        # friction velocity above the turbine
        self.ustar_hi = self.ustar * np.log( delta/self.z0_lo ) / np.log( delta/self.z0_hi )

        # friction velocity below the turbine
        self.ustar_lo = self.ustar_hi * np.log( self.zh/self.z0_hi * ( 1 + self.R / self.zh ) ** beta) / np.log( self.zh / self.z0_lo * ( 1-self.R / self.zh ) ** beta)

        # mean velocity at hub height and when applied individually to each cell, the mean velocity is given by
        self.uh = self.ustar_hi / self.kappa * np.log( self.zh / self.z0_hi * ( 1 + self.R / self.zh ) ** beta)

    def calculate_area(self):

        # Find region each turbine owns
        s_big = self.expand_array(self.s)
        s_big = self.expand_array(np.array(s_big))

        D = [self.fi.floris.farm.turbines[i].rotor_diameter for i in range(self.N)]

        # Determine initial trip line for IBL development based on convex hull
        hull = spt.ConvexHull(self.s)
        outside_N = len(hull.vertices)
        new_v = np.zeros((outside_N, 2))
        self.center = np.zeros((outside_N, 2))
        self.center[:, 0] = (max(self.s[hull.vertices, 0]) - min(self.s[hull.vertices, 0])) / 2 + min(
            self.s[hull.vertices, 0])
        self.center[:, 1] = (max(self.s[hull.vertices, 1]) - min(self.s[hull.vertices, 1])) / 2 + min(
            self.s[hull.vertices, 1])
        xdis = self.center[:, 0] - self.s[hull.vertices, 0]
        ydis = self.center[:, 1] - self.s[hull.vertices, 1]

        L = np.sqrt(ydis ** 2 + xdis ** 2)
        aprime = (D[0:outside_N] / L) * ydis
        bprime = (D[0:outside_N] / L) * xdis
        new_v[:, 0] = self.s[hull.vertices, 0] - bprime
        new_v[:, 1] = self.s[hull.vertices, 1] - aprime
        front_half = new_v[new_v[:, 0] < self.center[:, 0]]
        f2 = interp.interp1d(front_half[:, 1], front_half[:, 0], kind='linear', fill_value='extrapolate')

        self.xo = np.zeros(self.N)
        self.xo = f2(self.s[:, 1])

        # Use expanded voronoi diagram to extract polygons
        vor = spt.Voronoi(s_big)
        self.polys = [None] * self.N
        for (p, r) in zip(vor.points, vor.point_region):
            ind = np.where(np.logical_and(self.s[:, 0] == p[0], self.s[:, 1] == p[1]))[0]
            if (len(ind) == 1):
                self.polys[ind[0]] = geom.Polygon(vor.vertices[vor.regions[r]])

        # Determine which polygons are in front of each cell
        self.polys_crossed = [None] * self.N
        self.ind = np.zeros((self.N))
        for i in range(self.N):
            lineseg = geom.LineString([[self.s[i, 0], self.s[i, 1]], [self.xo[i] + D[i], self.s[i, 1]]])
            self.polys_crossed[i] = [lineseg.intersects(kj) for kj in self.polys]

            # Find the grid points that corresponds to each turbine
            self.Apf = np.zeros(self.N)
            for i in range(0, self.N):
                self.Apf[i] = self.polys[i].area

    def farm_thrust_coefficient(self):

        # compute the farm thrust coefficient
        uh = np.zeros(self.N)
        fbar = np.zeros(self.N)
        Ctp = np.zeros(self.N)
        Ct = np.zeros(self.N)
        for i in range(self.N):
            uh[i] = self.fi.floris.farm.turbines[i].average_velocity
            Ct[i] = self.fi.floris.farm.turbines[i].Ct
            Ctp[i] = 2 * (self.Uinfty / self.fi.floris.farm.turbines[i].average_velocity) * ( 1 - np.sqrt(1 - Ct[i]))
            fbar[i] = 0.5 * Ctp[i] * self.fi.floris.farm.turbines[i].average_velocity ** 2 * 0.25 \
                      * np.pi * self.fi.floris.farm.turbines[i].rotor_diameter ** 2

        cft = np.zeros(self.N)

        # XXX TODO: rotated_x and rotated_y from the floris code
        rotated_x = copy.deepcopy(np.array(self.layout_x))
        rotated_y = copy.deepcopy(np.array(self.layout_y))

        # determine which turbines influence the current turbine
        for i in range(self.N):
            # strTurb = 'Turbine ' + str(i)
            # plt.text(rotated_x[i],rotated_y[i],strTurb)
            idx = np.where( ((rotated_x - rotated_x[i]) <= 0) & (np.abs(rotated_y - rotated_y[i]) < self.fi.floris.farm.turbines[i].rotor_diameter) )
            # print('Turbine: ', i, 'Upstream turbines: ', idx)
            # if i == 20:
            #     for j in range(len(idx[0])):
            #         plt.plot([rotated_x[i],rotated_x[idx[0][j]]],[rotated_y[i],rotated_y[idx[0][j]]])
            # cft[i] = 2 * sum(fbar[idx]) / sum(self.Apf[idx] * uh[idx] ** 2)
            cft[i] = 2 * sum(fbar[self.polys_crossed[i]]) / sum(self.Apf[self.polys_crossed[i]] * uh[self.polys_crossed[i]] ** 2)

        return cft

    def expand_array(self, s):

        # expand the voronoi cells

        # First, find the interior points
        vor = spt.Voronoi(s)
        polys = [None] * np.size(s, 0)
        isInterior = [False] * np.size(s, 0)
        for i, (p, r) in enumerate(zip(vor.points, vor.point_region)):
            reg = vor.regions[r]
            if np.sum(np.array(reg) < 0):
                point = None
                polys[i] = None
            else:
                if sum(np.array(reg) > 0) > 2:
                    point = geom.Point(p)
                    polys[i] = geom.Polygon(vor.vertices[reg])
                    if point.within(polys[i]):
                        if self.aspect_ratio(polys[i].exterior.xy) > 5:
                            point = None
                            polys[i] = None
                        else:
                            isInterior[i] = True
                    else:
                        point = None
                        polys[i] = None

        # Expand the array so that we can define edges
        s_big = np.array(s).tolist()
        for (p, r) in zip(vor.ridge_points, vor.ridge_vertices):
            if np.logical_and(not (isInterior[p[1]]), isInterior[p[0]]):
                interior_pt = p[0]
                exterior_pt = p[1]
                xi = s[interior_pt, 0]
                yi = s[interior_pt, 1]
                xe = s[exterior_pt, 0]
                ye = s[exterior_pt, 1]
                if xe == xi:
                    xo = xe
                    yo = 2 * ye - yi
                elif ye == yi:
                    yo = ye
                    xo = 2 * xe - xi
                else:
                    m = (ye - yi) / (xe - xi)
                    xo = 2 * xe - xi
                    yo = ye + m * (xe - xi)
                s_big.append([xo, yo])

            if np.logical_and(not (isInterior[p[0]]), isInterior[p[1]]):
                interior_pt = p[1]
                exterior_pt = p[0]
                xi = s[interior_pt, 0]
                yi = s[interior_pt, 1]
                xe = s[exterior_pt, 0]
                ye = s[exterior_pt, 1]
                if xe == xi:
                    xo = xe
                    yo = 2 * ye - yi
                elif ye == yi:
                    yo = ye
                    xo = 2 * xe - xi
                else:
                    m = (ye - yi) / (xe - xi)
                    xo = 2 * xe - xi
                    yo = ye + m * (xe - xi)
                s_big.append([xo, yo])

        return np.unique(np.array(s_big), axis=0)

    def aspect_ratio(self, A):
        _, v = np.linalg.eigh(np.cov(A))
        rot = np.angle(v[0][0] + 1j * v[0][1])
        At = np.zeros(np.shape(A))
        for i in range(0, np.size(A, 1)):
            At[0][i] = A[0][i] * np.cos(rot) + A[1][i] * np.sin(rot)
            At[1][i] = -A[0][i] * np.sin(rot) + A[1][i] * np.cos(rot)
        dx = np.max(At[0]) - np.min(At[0])
        dy = np.max(At[1]) - np.min(At[1])
        AR = np.max([dx, dy]) / np.min([dx, dy])
        return AR

