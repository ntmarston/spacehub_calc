import pandas as pd
import numpy as np
import numbers
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.animation as animation
from astropy import units as u
from astropy.constants import G, c
import matplotlib as mpl
mpl.rcParams['animation.embed_limit'] = 250
from astropy.units import Quantity, UnitTypeError
from scipy.interpolate import PchipInterpolator
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

class TwoBodyOrbit:
    """
    Represents a two-body orbital system and computes Keplerian orbital elements.

    Parses SpaceHub simulation output and calculates orbital parameters including
    semi-major axis, eccentricity, inclination, and anomalies. Expects deviations
    when using non-conservative methods (e.g. Post-Newtonian approximation).

    @param filename: Path to the SpaceHub CSV output file.
    @param i: Index of the primary mass object (default 0).
    @param j: Index of the secondary mass object (default 1).

    @attr npoints: Number of timesteps in the simulation.
    @attr data: Pandas DataFrame containing raw simulation data.
    @attr M_i: Mass of the primary object.
    @attr M_j: Mass of the secondary object.
    @attr time: List of simulation timestamps.
    @attr R_vec: Relative position vector [X, Y, Z] components.
    @attr V_vec: Relative velocity vector [vx, vy, vz] components.
    @attr eccentricity: Scalar eccentricity at each timestep.
    @attr semiMajorAxis: Semi-major axis at each timestep (AU).
    @attr inclination_deg: Orbital inclination in degrees.
    @attr LongitudeAscendingNode_deg: Longitude of ascending node in degrees.
    @attr argument_of_periapsis_deg: Argument of periapsis in degrees.
    @attr true_anomaly_deg: True anomaly in degrees.
    @attr eccentric_anomaly_deg: Eccentric anomaly in degrees.
    """

    npoints = 0
    data = None
    i = 0 #Primary mass object
    j = 1 #Secondary mass object

    M_i = None
    M_j = None


    #Orbit State Vectors
    time = None
    R_vec = None
    V_vec = None
    magR = None
    hvec = None
    e_vec = None
    N_vec = None
    magN = None

    eccentricity = None
    semiMajorAxis = None
    inclination_rad = None
    inclination_deg = None


    LongitudeAscendingNode = None
    LongitudeAscendingNode_deg = None
    sinOmega = None
    cosOmega = None

    true_anomaly_rad = None
    true_anomaly_deg = None
    eccentric_anomaly_rad = None
    eccentric_anomaly_deg = None
    sinf = None
    cosf = None

    argument_of_periapsis_rad = None
    argument_of_periapsis_deg = None

    time_of_pericenter_passage = None
    orbital_period = None

    c0 = None
    points_per_orbit = None
    points_per_orbit_avg = None

    #Setters and Update methods

    def set_data(self, value):
        """
        Sets the simulation data after dropping NaN values.

        @param value: Pandas DataFrame containing simulation output.
        """
        self.data = value.dropna()

    def set_npoints(self):
        """
        Determines the number of valid timesteps by taking the minimum
        count between particles i and j. Sets self.npoints.
        """
        timesteps0 = len(self.data[self.data["id"]==self.i]["time"])
        timesteps1 = len(self.data[self.data["id"]==self.j]["time"])
        self.npoints = min(timesteps0, timesteps1)

    def set_time(self):
        """
        Extracts the time column from the particle with fewer timesteps
        to ensure array alignment. Sets self.time as a list.
        """
        timesteps_i = len(self.data[self.data["id"]==self.i]["time"])
        timesteps_j = len(self.data[self.data["id"]==self.j]["time"])
        if timesteps_i < timesteps_j:
            col = self.data[self.data["id"]==self.i]["time"]
        else: #If they have the same number of timesteps, or if j has more
            col = self.data[self.data["id"]==self.j]["time"]


        self.time = col.to_list()

    def set_ij(self, i, j):
        """
        Sets the primary and secondary particle indices.

        @param i: Index of the primary mass object.
        @param j: Index of the secondary mass object.
        """
        self.i = i
        self.j = j

    def set_masses(self):
        """
        Retrieves and stores masses for particles i and j.
        Sets self.M_i and self.M_j.
        """
        mi = get_tot_mass(self.data, self.i)
        mj = get_tot_mass(self.data, self.j)
        self.M_i = mi
        self.M_j = mj

    def set_R_and_V(self):
        """
        Computes relative position and velocity vectors between particles i and j.
        Sets self.R_vec, self.magR, and self.V_vec.
        """
        data, i, j = self.data, self.i, self.j
        X, Y, Z = distance(data, 'p', i, j, self.npoints)
        magR = mag([X, Y, Z])
        self.R_vec = [X, Y, Z]
        self.magR = magR

        vx, vy, vz = distance(data, 'v', i, j, self.npoints)
        self.V_vec = [vx, vy, vz]

    def set_h_vector(self):
        """
        Computes the specific angular momentum vector h = r x v.
        Sets self.hvec as [hx, hy, hz] component lists.
        Requires set_R_and_V() to be called first.
        """
        self.hvec = calc_h_vector(self.R_vec, self.V_vec)
   
    def set_ecc_vector(self):
        """
        Computes the eccentricity vector at each timestep.
        Sets self.e_vec as [ex, ey, ez] arrays.
        Requires set_R_and_V() and set_masses() to be called first.
        """
        m_tot = self.M_i + self.M_j
        dx, dy, dz = np.asarray(self.R_vec[0]), np.asarray(self.R_vec[1]), np.asarray(self.R_vec[2])
        dvx, dvy, dvz = np.asarray(self.V_vec[0]), np.asarray(self.V_vec[1]), np.asarray(self.V_vec[2])

        # Vectorized: calc_ecc already supports array operations
        ex, ey, ez = calc_ecc(m_tot, dx, dy, dz, dvx, dvy, dvz)
        self.e_vec = [ex, ey, ez]

    def set_N_vector(self):
        """
        Computes the node vector N = k_hat x h at each timestep.
        The node vector points toward the ascending node.
        Sets self.N_vec and self.magN.
        Requires set_h_vector() to be called first.
        """
        # Vectorized: stack h-vector components and use np.cross on full arrays
        hx, hy, hz = np.asarray(self.hvec[0]), np.asarray(self.hvec[1]), np.asarray(self.hvec[2])
        h_arr = np.column_stack([hx, hy, hz])
        khat = np.array([0, 0, 1])

        # np.cross broadcasts khat across all rows
        N_arr = np.cross(khat, h_arr)

        self.N_vec = [N_arr[:, 0], N_arr[:, 1], N_arr[:, 2]]
        self.magN = np.sqrt(N_arr[:, 0]**2 + N_arr[:, 1]**2 + N_arr[:, 2]**2)

    rdebug = []
    v2debug = []

    def set_sma(self):
        """
        Computes the semi-major axis at each timestep using the vis-viva equation.
        Based on Murray-Dermott Eq. 2.134. Sets self.semiMajorAxis in AU.
        Also populates rdebug and v2debug arrays for debugging.
        Requires set_R_and_V() and set_masses() to be called first.
        """
        m_tot = self.M_i + self.M_j
        dx, dy, dz = np.asarray(self.R_vec[0]), np.asarray(self.R_vec[1]), np.asarray(self.R_vec[2])
        dvx, dvy, dvz = np.asarray(self.V_vec[0]), np.asarray(self.V_vec[1]), np.asarray(self.V_vec[2])

        # Vectorized: calc_sma already supports array operations
        self.semiMajorAxis = calc_sma(m_tot, dx, dy, dz, dvx, dvy, dvz)
        self.rdebug = np.sqrt(dx**2 + dy**2 + dz**2)
        self.v2debug = dvx**2 + dvy**2 + dvz**2

    def set_scalar_e(self):
        """
        Computes scalar eccentricity from angular momentum and semi-major axis.
        Uses Murray-Dermott Eq. 2.135: e = sqrt(1 - h^2/(mu*a)).
        Sets self.eccentricity as a NumPy array.
        Requires set_masses(), set_h_vector(), and set_sma() to be called first.
        """
        mu = self.M_i + self.M_j
        h = np.asarray(mag(self.hvec))
        a = np.asarray(self.semiMajorAxis)
        self.eccentricity = np.sqrt(1 - (h**2) / (mu * a))

    def set_inclination(self):
        """
        Computes orbital inclination as the angle between h-vector and z-axis.
        Uses i = arccos(h_z / |h|). Sets self.inclination_rad and self.inclination_deg.
        Requires set_h_vector() to be called first.
        """
        hz = np.asarray(self.hvec[2])
        h = np.asarray(mag(self.hvec))
        self.inclination_rad = np.arccos(hz / h)
        self.inclination_deg = np.rad2deg(self.inclination_rad)

    def set_longitude_of_ascending_node(self):
        """
        Computes the longitude of the ascending node (Omega).
        The angle is measured from the reference direction to the ascending node.
        Sets self.LongitudeAscendingNode (rad), self.LongitudeAscendingNode_deg,
        self.sinOmega, and self.cosOmega as NumPy arrays.
        Requires set_h_vector() and set_inclination() to be called first.

        @throws AssertionError: If sin^2 + cos^2 deviates significantly from 1.
        """
        hx_raw = np.asarray(self.hvec[0])
        hy_raw = np.asarray(self.hvec[1])
        hz = np.asarray(self.hvec[2])
        h = np.asarray(mag(self.hvec))
        incl = np.asarray(self.inclination_rad)

        # Conditional sign flip based on hz
        hx = np.where(hz > 0, hx_raw, -hx_raw)
        hy = np.where(hz > 0, -hy_raw, hy_raw)

        denom = h * np.sin(incl)
        sines = hx / denom
        cosines = hy / denom

        Omegas = np.arcsin(sines)
        # Handle NaN values with arccos fallback (negative sign is a hotfix from original)
        Omegas = np.where(np.isnan(Omegas), -np.arccos(cosines), Omegas)
        # Shift negative values to positive range
        Omegas = np.where(Omegas < 0, 2*np.pi + Omegas, Omegas)

        # Checkpoint validation (skip where inclination is near zero)
        valid_mask = np.abs(incl) >= 0.01
        checksum = sines[valid_mask]**2 + cosines[valid_mask]**2
        violations = np.abs(1 - checksum) >= 0.1
        if np.any(violations):
            bad_idx = np.where(valid_mask)[0][np.where(violations)[0][0]]
            raise AssertionError(f"Checkpoint test failed in set_longitude_of_ascending_node. sin^2+cos^2 = {checksum[np.where(violations)[0][0]]} (pn: {bad_idx})")

        self.sinOmega = sines
        self.cosOmega = cosines
        self.LongitudeAscendingNode = Omegas
        self.LongitudeAscendingNode_deg = np.rad2deg(Omegas)
    
    def set_true_anomaly(self):
        """
        Computes the true anomaly using the eccentricity vector method.
        True anomaly f is the angle between periapsis and the current position.
        Uses sign of r.v to determine quadrant.
        Sets self.true_anomaly_rad, self.true_anomaly_deg, self.sinf, self.cosf as NumPy arrays.
        """
        ex, ey, ez = np.asarray(self.e_vec[0]), np.asarray(self.e_vec[1]), np.asarray(self.e_vec[2])
        rx, ry, rz = np.asarray(self.R_vec[0]), np.asarray(self.R_vec[1]), np.asarray(self.R_vec[2])
        vx, vy, vz = np.asarray(self.V_vec[0]), np.asarray(self.V_vec[1]), np.asarray(self.V_vec[2])

        edotr = ex*rx + ey*ry + ez*rz
        mag_e = np.sqrt(ex**2 + ey**2 + ez**2)
        mag_r = np.sqrt(rx**2 + ry**2 + rz**2)
        rdotv = rx*vx + ry*vy + rz*vz

        # Clip cosf to [-1, 1] to handle precision issues
        cosf = np.clip(edotr / (mag_e * mag_r), -1, 1)

        # Determine quadrant based on sign of r.v
        f = np.where(rdotv > 0, np.arccos(cosf), 2*np.pi - np.arccos(cosf))
        sinf = np.sqrt(1 - cosf**2)

        self.true_anomaly_rad = f
        self.true_anomaly_deg = np.rad2deg(f)
        self.sinf = sinf
        self.cosf = cosf

    def set_argument_of_periapsis(self):
        """
        Computes the argument of periapsis (omega).
        The angle from the ascending node to periapsis, measured in the orbital plane.
        Uses sign of e_z to determine quadrant.
        Sets self.argument_of_periapsis_rad and self.argument_of_periapsis_deg as NumPy arrays.
        """
        nx, ny, nz = np.asarray(self.N_vec[0]), np.asarray(self.N_vec[1]), np.asarray(self.N_vec[2])
        ex, ey, ez = np.asarray(self.e_vec[0]), np.asarray(self.e_vec[1]), np.asarray(self.e_vec[2])
        N_mag = np.asarray(self.magN)
        e_scalar = np.asarray(self.eccentricity)

        Ndote = nx*ex + ny*ey + nz*ez
        cos_omega = Ndote / (N_mag * e_scalar)
        omega = np.where(ez >= 0, np.arccos(cos_omega), 2*np.pi - np.arccos(cos_omega))

        self.argument_of_periapsis_rad = omega
        self.argument_of_periapsis_deg = np.rad2deg(omega)

    def set_eccentric_anomaly(self):
        """
        Computes the eccentric anomaly E from orbital elements.
        Uses Murray-Dermott Eq. 2.42: r = a(1 - e*cos(E)).
        Uses sign of r.v to determine quadrant.
        Sets self.eccentric_anomaly_rad and self.eccentric_anomaly_deg as NumPy arrays.
        """
        rx, ry, rz = np.asarray(self.R_vec[0]), np.asarray(self.R_vec[1]), np.asarray(self.R_vec[2])
        vx, vy, vz = np.asarray(self.V_vec[0]), np.asarray(self.V_vec[1]), np.asarray(self.V_vec[2])
        r = np.asarray(self.magR)
        e = np.asarray(self.eccentricity)
        a = np.asarray(self.semiMajorAxis)

        rdotv = rx*vx + ry*vy + rz*vz
        arg = (1/e) * (-r/a + 1)
        E = np.where(rdotv > 0, np.arccos(arg), 2*np.pi - np.arccos(arg))

        self.eccentric_anomaly_rad = E
        self.eccentric_anomaly_deg = np.rad2deg(E)

    def set_time_of_pericenter_passage(self):
        """
        Computes the time of pericenter passage (tau) using Kepler's equation.
        tau = t - (E - e*sin(E)) / n, where n = sqrt(mu/a^3).
        Sets self.time_of_pericenter_passage as a NumPy array.

        Requires set_masses() to be called first.
        """
        t = np.asarray(self.time)
        E = np.asarray(self.eccentric_anomaly_rad)
        e = np.asarray(self.eccentricity)
        a = np.asarray(self.semiMajorAxis)
        mu = self.M_i + self.M_j
        self.time_of_pericenter_passage = t - (E - e*np.sin(E)) / np.sqrt(mu * a**(-3))

    def set_orbital_period(self, G=1):
        """
        Computes orbital period using Kepler's third law: P = 2*pi*sqrt(a^3/(G*M)).
        WARNING: This calculation has not been fully validated.

        @param G: Gravitational constant (default 1 for code units).
        """
        a = np.asarray(self.semiMajorAxis)
        self.orbital_period = 2*np.pi * np.sqrt(a**3 / (G * self.M_i))

    def set_c0(self, deviation_check=True):
        """
        Computes the Peters (1964) constant c0 from Eq. 5.48 for gravitational wave decay.
        c0 = a*(1-e^2)*e^(-12/19)*(1 + 121/304*e^2)^(-870/2299).
        Primarily for debugging PN2.5 simulations.

        @param deviation_check: If True, prints the range of c0 values (should be constant).
        """
        a = np.asarray(self.semiMajorAxis)
        e = np.asarray(self.eccentricity)
        c0 = a * (1 - e**2) * (e**(-12/19)) * (1 + (121/304)*e**2)**(-870/2299)
        if deviation_check:
            amplitude = np.max(c0) - np.min(c0)
            print(f"Calculated c0 values are within {amplitude:.4} of constant (maximum - minimum)")
        self.c0 = c0

    def set_points_per_orbit(self):
        """
        Calculates the number of simulation points per orbit by detecting
        wrap-arounds in true anomaly. Sets self.points_per_orbit (array)
        and self.points_per_orbit_avg (scalar mean).
        """
        df = np.diff(self.true_anomaly_deg)
        indices = np.where(np.sign(df) < 0)[0]
        extremes_t = [self.time[i] for i in indices]
        ppo = np.diff(indices)
        self.points_per_orbit = ppo
        self.points_per_orbit_avg = np.mean(ppo)

    # Mapping of convenient parameter names to attribute names
    PARAM_NAMES = {
        'a': 'semiMajorAxis',
        'sma': 'semiMajorAxis',
        'semi_major_axis': 'semiMajorAxis',
        'e': 'eccentricity',
        'ecc': 'eccentricity',
        'i': 'inclination_deg',
        'inc': 'inclination_deg',
        'inclination': 'inclination_deg',
        'R': 'magR',
        'r': 'magR',
        'separation': 'magR',
        'Omega': 'LongitudeAscendingNode_deg',
        'omega': 'argument_of_periapsis_deg',
        'f': 'true_anomaly_deg',
        'E': 'eccentric_anomaly_deg',
    }

    def find_crossing(self, param, threshold, direction='any', find_all=False, npoints=1e4, years=False):
        """
        Finds the time(s) at which an orbital parameter crosses a specified threshold.

        Uses PCHIP interpolation to find precise crossing times between simulation
        timesteps. Can detect crossings from above, from below, or in either direction.

        @param param: Orbital parameter to check. Can be:
                      - Attribute name: 'semiMajorAxis', 'eccentricity', 'inclination_deg', etc.
                      - Shorthand: 'a', 'e', 'i', 'R', 'Omega', 'omega', 'f', 'E'
        @param threshold: Target value to find crossing(s) at.
        @param direction: Which crossing direction to detect:
                          - 'any': Either direction (default)
                          - 'rising' or 'from_below': Parameter increasing through threshold
                          - 'falling' or 'from_above': Parameter decreasing through threshold
        @param find_all: If True, returns all crossings. If False, returns only the first (default False).
        @param npoints: Number of interpolation points (default 10000).
        @param in_years: If True, return time in years. If False, return in SpaceHub units (default False).
        @return: If find_all=False: (time, value) tuple of first crossing, or (None, None) if not found.
                 If find_all=True: List of (time, value) tuples for all crossings.

        Example usage:
            # Find when semi-major axis first crosses 0.8 AU
            t, val = orb.find_crossing('a', 0.8)

            # Find when eccentricity first crosses 0.5 from below
            t, val = orb.find_crossing('e', 0.5, direction='rising')

            # Find all times when separation crosses 1.0 AU
            crossings = orb.find_crossing('R', 1.0, find_all=True)
        """
        # Resolve parameter name
        attr_name = self.PARAM_NAMES.get(param, param)
        if not hasattr(self, attr_name):
            raise ValueError(f"Unknown parameter '{param}'. Available: {list(self.PARAM_NAMES.keys())} or any attribute name.")

        y = np.asarray(getattr(self, attr_name))
        x = np.asarray(self.time)

        # Interpolate
        f = PchipInterpolator(x, y)
        x_fine = np.linspace(min(x), max(x), int(npoints))
        y_fine = f(x_fine)

        # Find crossings by detecting sign changes in (y - threshold)
        offset = y_fine - threshold
        sign_changes = np.diff(np.sign(offset))

        # Determine which sign changes to consider based on direction
        if direction in ('rising', 'from_below'):
            # Sign changes from negative to positive (crossing upward)
            idx_list = np.argwhere(sign_changes > 0).flatten()
        elif direction in ('falling', 'from_above'):
            # Sign changes from positive to negative (crossing downward)
            idx_list = np.argwhere(sign_changes < 0).flatten()
        else:  # 'any'
            idx_list = np.argwhere(sign_changes != 0).flatten()

        if len(idx_list) == 0:
            if find_all:
                return []
            return (None, None)

        scale = 1 / (2 * np.pi) if years else 1

        if find_all:
            return [(x_fine[idx] * scale, y_fine[idx]) for idx in idx_list]
        else:
            idx = idx_list[0]
            return (x_fine[idx] * scale, y_fine[idx])

    def __init__(self, filename, i=0, j=1, mute=False):
        """
        Initializes a TwoBodyOrbit object from SpaceHub simulation output.
        Automatically computes all orbital elements upon construction.

        @param filename: Path to the SpaceHub CSV output file.
        @param i: Index of the primary mass object (default 0).
        @param j: Index of the secondary mass object (default 1).
        @param mute: If True, suppresses print output during initialization (default False).
        """
        self.data = load_spacehub_data(filename)
        self.i = i
        self.j = j
        if not mute:
            print("load data complete")
            print("Determining timesteps...")
        self.set_npoints()
        self.set_time()
        self.years = np.asarray(self.time) / (2 * np.pi)
        if not mute:
            print("Calculating orbital state vectors...")
        self.set_R_and_V()
        self.set_masses()
        self.set_h_vector()
        self.set_N_vector()
        self.set_ecc_vector()
        if not mute:
            print("Calculating scalar orbital elements...")
        self.set_sma()
        self.set_scalar_e()
        self.set_inclination()
        self.set_longitude_of_ascending_node()
        self.set_true_anomaly()
        self.set_argument_of_periapsis()
        self.set_eccentric_anomaly()
        self.set_time_of_pericenter_passage()
        self.set_points_per_orbit()
        self.set_c0()
        if not mute:
            print("WARNING: Orbital period calculation has not been checked")
        self.set_orbital_period()
        if not mute:
            print("Done")
    
    #===============
    #Output methods
    #===============

    def __str__(self):
        """
        Returns a string representation of initial orbital conditions.

        @return: Formatted string with t=0 orbital elements.
        """
        outstr = ""
        outstr += (f"Initial (t=0) Conditions:" + "\n"
                   + f"a: {self.semiMajorAxis[0]}AU" + "\n"
                   + f"e: {self.eccentricity[0]}" + "\n"
                   + f"i: {self.inclination_deg[0]}deg" + "\n"
                   + f"Longtiude of Ascending Node: {self.LongitudeAscendingNode_deg[0]}deg" + "\n"
                   + f"Argument of Periapsis: {self.argument_of_periapsis_deg[0]}deg" + "\n"
                   + f"True Anomaly: {self.true_anomaly_deg[0]}deg" + "\n"
                   + f"Orbital Separation: {self.magR[0]}" + "\n")
        return outstr

    def to_pandas(self):
        """
        Exports orbital data to a Pandas DataFrame.

        @return: DataFrame with orbital elements at each timestep.
        @throws NotImplementedError: This method is not yet implemented.
        """
        col_names = ["id", "x", "y", "z", "vx", "vy", "vz", "arg_of_peri_deg", "inclination_deg", "magN", "magR", "a", "true_anomaly_deg"]
        raise NotImplementedError("Not Implemented")
        pass

    def plot_orbit_3panel(self):
        """
        Creates a 3-panel visualization of the orbit.
        Panel 1: 3D scatter plot of the orbit with coordinate axes.
        Panel 2: X-Y projection (face-on view).
        Panel 3: X-Z projection (edge-on view).

        @return: Tuple of (fig, [ax0, ax1, ax2]) matplotlib objects.
        """
        df, i, j = self.data, self.i, self.j
        fig = plt.figure(figsize=(15, 5))
        ax0 = fig.add_subplot(131, projection='3d')
        ax1 = fig.add_subplot(132)
        ax2 = fig.add_subplot(133)
        #Plot axis spines
        ax0.quiver(0, 0, 0, 1, 0, 0, color='black', arrow_length_ratio=0.1, linewidth=2, label='X-axis')
        ax0.quiver(0, 0, 0,  0, 1, 0, color='black', arrow_length_ratio=0.1, linewidth=2, label='Y-axis')
        ax0.quiver(0, 0, 0, 0, 0, 1, color='black', arrow_length_ratio=0.1, linewidth=2, label='Z-axis')
        #plot orbit of j around i
        ax0.scatter(self.R_vec[0], self.R_vec[1], self.R_vec[2], s=0.3, c='red', zorder=2, label="Secondary")
        ax0.scatter(0, 0, 0, s=3, c='blue', zorder=1, label="Primary")
        ax1.scatter(0, 0, s=3, c='blue', zorder=1, label="Primary")
        ax1.scatter(self.R_vec[0], self.R_vec[1], s=0.3, c='red', zorder=2, label="Secondary")
        ax1.grid(visible=True, zorder=-1)
        ax1.set_xlim(-1,1)
        ax1.set_ylim(-1,1)

        ax2.scatter(self.R_vec[0], self.R_vec[2], s=0.3, c='red', zorder=2, label="Secondary")
        ax2.scatter(0, 0, s=3, c='blue', zorder=1, label="Primary")
        ax2.grid(visible=True, zorder=-1)

        return fig, [ax0, ax1, ax2]
    
    # Available plot types for plot_keplerian_evolution
    PLOT_CONFIG = {
        'e':     {'data': 'eccentricity',              'title': 'Eccentricity',                   'ylabel': r'$e$',            'ylim': (-0.1, 1)},
        'a':     {'data': 'semiMajorAxis',             'title': 'Semi-major Axis',                'ylabel': r'$a$ (AU)',       'ylim': None, 'fmt': '%.2f'},
        'i':     {'data': 'inclination_deg',           'title': 'Inclination',                    'ylabel': r'$i$ (deg)',      'ylim': (0, 360)},
        'R':     {'data': 'magR',                      'title': 'Separation',                     'ylabel': r'$||R||$ (AU)',   'ylim': (0, 10)},
        'Omega': {'data': 'LongitudeAscendingNode_deg','title': 'Longitude of Ascending Node',    'ylabel': r'$\Omega$ (deg)', 'ylim': (0, 360)},
        'omega': {'data': 'argument_of_periapsis_deg', 'title': 'Argument of Periapsis',          'ylabel': r'$\omega$ (deg)', 'ylim': (0, 360)},
        'f':     {'data': 'true_anomaly_deg',          'title': 'True Anomaly',                   'ylabel': r'$f$ (deg)',      'ylim': (0, 360)},
        'E':     {'data': 'eccentric_anomaly_deg',     'title': 'Eccentric Anomaly',              'ylabel': r'$E$ (deg)',      'ylim': (0, 360)},
    }

    def plot_keplerian_evolution(self, plots=('e', 'a', 'i', 'R'), xlim=None, ylim=None, figsize=None):
        """
        Plots up to 4 orbital elements in a grid.
        
        @param plots: Tuple/list of element keys to plot (1-4 elements).
                      Options: 'e' (eccentricity), 'a' (semi-major axis),
                      'i' (inclination), 'R' (separation), 'Omega' (longitude
                      of ascending node), 'omega' (argument of periapsis),
                      'f' (true anomaly), 'E' (eccentric anomaly).
        @param xlim: Tuple (min, max) for x-axis limits. Defaults to full time range.
        @param ylim: Controls y-axis scaling:
                     - None (default): use preset limits from PLOT_CONFIG
                     - False: auto-scale all plots (matplotlib default)
                     - (min, max) tuple: apply same limits to all plots
                     - List of tuples/None/False: per-plot limits, where
                       None uses preset and False uses auto-scale
        @param figsize: Tuple (width, height) for figure size. Defaults vary by layout.
        @return: Tuple of (fig, axs) where axs is a list of Axes objects.
        """
        plots = list(plots)[:4]
        n = len(plots)
        xlim = xlim or (0, max(self.time))

        # Default figure sizes
        default_sizes = {1: (6, 5), 2: (12, 5), 3: (12, 8), 4: (12, 8)}
        figsize = figsize or default_sizes[n]

        # Normalize ylim to a list of per-plot values
        if ylim is None:
            # Use presets for all
            ylim_list = [None] * n
        elif ylim is False:
            # Auto-scale all
            ylim_list = [False] * n
        elif isinstance(ylim, list):
            # Per-plot specification
            ylim_list = ylim + [None] * (n - len(ylim))
        else:
            # Single tuple applies to all
            ylim_list = [ylim] * n

        # Create appropriate grid layout
        if n == 1:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            axs = [ax]
        elif n == 2:
            fig, axs = plt.subplots(1, 2, figsize=figsize)
            axs = list(axs)
        elif n == 3:
            fig = plt.figure(figsize=figsize)
            axs = [
                fig.add_subplot(2, 2, 1),
                fig.add_subplot(2, 2, 2),
                fig.add_subplot(2, 1, 2),  # Bottom row spans both columns
            ]
        else:  # n == 4
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            axs = list(axes.flatten())

        # Plot each element
        for idx, (ax, key) in enumerate(zip(axs, plots)):
            cfg = self.PLOT_CONFIG[key]
            ax.plot(self.time, getattr(self, cfg['data']))
            ax.set_xlim(xlim)
            ax.set_title(cfg['title'])
            ax.set_ylabel(cfg['ylabel'])
            ax.set_xlabel(r'$yr (2\pi)^{-1}$')

            # Determine ylim for this plot
            plot_ylim = ylim_list[idx]
            if plot_ylim is None:
                # Use preset from config
                if cfg['ylim']:
                    ax.set_ylim(cfg['ylim'])
            elif plot_ylim is not False:
                # Use custom limits (False means auto-scale, so do nothing)
                ax.set_ylim(plot_ylim)

            if cfg.get('fmt'):
                ax.yaxis.set_major_formatter(FormatStrFormatter(cfg['fmt']))

        fig.tight_layout()
        return fig, axs

    def plot_keplerian_evolution_basic(self, xlim=None, time_stop=0):
        """
        Plots all 6 standard orbital elements in a 2x3 grid layout.
        Legacy method retained for backward compatibility.

        Elements plotted: eccentricity, inclination, separation,
        longitude of ascending node, semi-major axis, argument of periapsis.

        @param xlim: Tuple (min, max) for x-axis limits. Defaults to full time range.
        @param time_stop: Unused parameter (retained for backward compatibility).
        @return: Tuple of (fig, axs) where axs is a flattened array of 6 Axes.
        """
        xlim = xlim or (0, max(self.time))
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axs = axes.flatten()
        plot_keys = ['e', 'i', 'R', 'Omega', 'a', 'omega']

        for ax, key in zip(axs, plot_keys):
            cfg = self.PLOT_CONFIG[key]
            ax.plot(self.time, getattr(self, cfg['data']))
            ax.set_xlim(xlim)
            ax.set_title(cfg['title'])
            ax.set_ylabel(cfg['ylabel'])
            ax.set_xlabel(r'$yr (2\pi)^{-1}$')
            if cfg['ylim']:
                ax.set_ylim(cfg['ylim'])
            if cfg.get('fmt'):
                ax.yaxis.set_major_formatter(FormatStrFormatter(cfg['fmt']))

        fig.tight_layout()
        return fig, axs

    def plot_trajectory_3d(self, fig, ax, start_index=0, *args, **kwargs):
        """
        Creates an animated 3D trajectory visualization of the orbit.

        Runtime may be long if frames and interval are not specified.
        Use HTML(ani.to_jshtml()) to render in IPython Notebooks.

        Example usage:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ani = orb.plot_trajectory_3d(fig, ax, frames=500, interval=20)
            HTML(ani.to_jshtml())

        @param fig: Matplotlib figure object to write the animation to.
        @param ax: Matplotlib 3D axes object for plotting.
        @param start_index: Starting frame index in the data (default 0).
        @param args: Additional positional arguments passed to FuncAnimation.
        @param kwargs: Additional keyword arguments passed to FuncAnimation.
                       Common options: frames (int), interval (int, ms between frames).
        @return: matplotlib.animation.FuncAnimation object.
        """

        rx = self.R_vec[0][start_index:]
        ry = self.R_vec[1][start_index:]
        rz = self.R_vec[2][start_index:]
        global traj_points
        traj_points = ax.scatter3D(rx[0], ry[0], rz[0], c="purple")
        ref_frame_particle = ax.scatter3D(0, 0, 0, c="blue")
        traj = ax.plot(rx[0], ry[0], rz[0], c="gold")[0]


        def update(frame_num):
            # for each frame, update the data stored on each artist.
            window_size = 1
            window_start = frame_num-window_size if (frame_num > window_size) else 0
            x = rx[window_start:frame_num]
            y = ry[window_start:frame_num]
            z = rz[window_start:frame_num]
            # Update scatter
            data = np.stack([x,y,z]).T
            global traj_points  # Need global to reassign
            traj_points.remove()
            traj_points = ax.scatter3D(x, y, z, c="blue")
            # update the line plot:
            traj.set_data([rx[:frame_num], ry[:frame_num]])
            traj.set_3d_properties(rz[:frame_num])

            #traj.set_zdata(z[:frame])
            return (traj_points, traj)
        
        ani = animation.FuncAnimation(fig=fig, func=update, *args, **kwargs)
        return ani





class NBodyVisualizer:
    """
    Visualizer for N-body SpaceHub simulation outputs.

    Loads and organizes particle data from SpaceHub CSV output for easy
    access and visualization. Supports arbitrary numbers of particles.

    @param filename: Path to the SpaceHub CSV output file.

    @attr n_particles: Number of particles in the simulation.
    @attr npoints: Number of timesteps in the simulation.
    @attr time: NumPy array of simulation timestamps.
    @attr masses: NumPy array of particle masses indexed by particle ID.
    @attr positions: Dict mapping particle ID to dict with 'x', 'y', 'z' arrays.
    @attr velocities: Dict mapping particle ID to dict with 'x', 'y', 'z' arrays.
    @attr data: Raw pandas DataFrame containing all simulation data.
    """

    def __init__(self, filename):
        """
        Initializes an NBodyVisualizer from SpaceHub simulation output.

        Loads the CSV file and organizes data by particle for easy access.
        Each particle's position and velocity components are stored as
        NumPy arrays indexed by timestep.

        @param filename: Path to the SpaceHub CSV output file.
        """
        # Load raw data
        self.data = load_spacehub_data(filename)

        # Determine number of particles
        self.n_particles = self.data["id"].nunique()
        particle_ids = sorted(self.data["id"].unique().astype(int))

        # Determine number of timesteps (use minimum across all particles)
        timesteps_per_particle = [
            len(self.data[self.data["id"] == pid]) for pid in particle_ids
        ]
        self.npoints = min(timesteps_per_particle)

        # Extract time array (from first particle)
        self.time = self.data[self.data["id"] == particle_ids[0]]["time"].values[:self.npoints]

        # Extract masses (one value per particle)
        self.masses = np.array([
            self.data[self.data["id"] == pid]["mass"].iloc[0] for pid in particle_ids
        ])

        # Organize positions by particle
        self.positions = {}
        for pid in particle_ids:
            particle_data = self.data[self.data["id"] == pid]
            self.positions[pid] = {
                'x': particle_data['px'].values[:self.npoints],
                'y': particle_data['py'].values[:self.npoints],
                'z': particle_data['pz'].values[:self.npoints],
            }

        # Organize velocities by particle
        self.velocities = {}
        for pid in particle_ids:
            particle_data = self.data[self.data["id"] == pid]
            self.velocities[pid] = {
                'x': particle_data['vx'].values[:self.npoints],
                'y': particle_data['vy'].values[:self.npoints],
                'z': particle_data['vz'].values[:self.npoints],
            }

        print(f"Loaded {self.n_particles} particles with {self.npoints} timesteps")

    def get_position(self, particle_id, timestep=None, relative_to=None):
        """
        Gets position vector(s) for a particle, optionally relative to another particle.

        @param particle_id: ID of the particle (0-indexed).
        @param timestep: Optional timestep index. If None, returns full time series.
        @param relative_to: Optional particle ID to compute position relative to.
                           Returns (particle_id position) - (relative_to position).
        @return: Position as [x, y, z] array (if timestep given) or dict with 'x', 'y', 'z' arrays.
        """
        pos = self.positions[particle_id]

        if relative_to is not None:
            ref_pos = self.positions[relative_to]
            if timestep is not None:
                return np.array([
                    pos['x'][timestep] - ref_pos['x'][timestep],
                    pos['y'][timestep] - ref_pos['y'][timestep],
                    pos['z'][timestep] - ref_pos['z'][timestep]
                ])
            return {
                'x': pos['x'] - ref_pos['x'],
                'y': pos['y'] - ref_pos['y'],
                'z': pos['z'] - ref_pos['z'],
            }

        if timestep is not None:
            return np.array([pos['x'][timestep], pos['y'][timestep], pos['z'][timestep]])
        return pos

    def get_velocity(self, particle_id, timestep=None, relative_to=None):
        """
        Gets velocity vector(s) for a particle, optionally relative to another particle.

        @param particle_id: ID of the particle (0-indexed).
        @param timestep: Optional timestep index. If None, returns full time series.
        @param relative_to: Optional particle ID to compute velocity relative to.
                           Returns (particle_id velocity) - (relative_to velocity).
        @return: Velocity as [vx, vy, vz] array (if timestep given) or dict with 'x', 'y', 'z' arrays.
        """
        vel = self.velocities[particle_id]

        if relative_to is not None:
            ref_vel = self.velocities[relative_to]
            if timestep is not None:
                return np.array([
                    vel['x'][timestep] - ref_vel['x'][timestep],
                    vel['y'][timestep] - ref_vel['y'][timestep],
                    vel['z'][timestep] - ref_vel['z'][timestep]
                ])
            return {
                'x': vel['x'] - ref_vel['x'],
                'y': vel['y'] - ref_vel['y'],
                'z': vel['z'] - ref_vel['z'],
            }

        if timestep is not None:
            return np.array([vel['x'][timestep], vel['y'][timestep], vel['z'][timestep]])
        return vel

    def get_mass(self, particle_id):
        """
        Gets the mass of a particle.

        @param particle_id: ID of the particle (0-indexed).
        @return: Mass of the particle.
        """
        return self.masses[particle_id]

    # Default color cycle for particle visualization
    DEFAULT_COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']

    def animate_trajectories(self, fig, ax, reference_frame=0, start_index=0,
                             tail_length=None, colors=None, sizes=None, *args, **kwargs):
        """
        Creates an animated 3D trajectory visualization for all particles.

        The animation shows particle positions evolving over time with trailing
        trajectory lines. The reference frame can be centered on a particle
        or fixed at a point in space.

        Use HTML(ani.to_jshtml()) to render in IPython Notebooks.

        Example usage:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            # Center on particle 0
            ani = viz.animate_trajectories(fig, ax, reference_frame=0, frames=500, interval=20)
            # Or center on a fixed point
            ani = viz.animate_trajectories(fig, ax, reference_frame=(0, 0, 0), frames=500)
            HTML(ani.to_jshtml())

        @param fig: Matplotlib figure object to write the animation to.
        @param ax: Matplotlib 3D axes object for plotting.
        @param reference_frame: Reference frame specification:
                               - int: Particle ID to center the view on (tracks that particle).
                               - tuple/list of 3 floats: Fixed point (x, y, z) in space.
                               Default is 0 (center on particle 0).
        @param start_index: Starting frame index in the data (default 0).
        @param tail_length: Number of frames for trajectory tail. None shows full trail.
        @param colors: List of colors for each particle. Uses DEFAULT_COLORS if None.
        @param sizes: List of marker sizes for each particle. Defaults to 20 for all.
        @param args: Additional positional arguments passed to FuncAnimation.
        @param kwargs: Additional keyword arguments passed to FuncAnimation.
                       Common options: frames (int), interval (int, ms between frames).
        @return: matplotlib.animation.FuncAnimation object.
        """
        particle_ids = list(self.positions.keys())
        n = len(particle_ids)

        # Set default colors and sizes
        if colors is None:
            colors = [self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)] for i in range(n)]
        if sizes is None:
            sizes = [20] * n

        # Compute positions relative to reference frame
        transformed_positions = {}
        for pid in particle_ids:
            px = self.positions[pid]['x'][start_index:]
            py = self.positions[pid]['y'][start_index:]
            pz = self.positions[pid]['z'][start_index:]

            if isinstance(reference_frame, int):
                # Center on a particle (subtract that particle's position)
                ref_x = self.positions[reference_frame]['x'][start_index:]
                ref_y = self.positions[reference_frame]['y'][start_index:]
                ref_z = self.positions[reference_frame]['z'][start_index:]
                transformed_positions[pid] = {
                    'x': px - ref_x,
                    'y': py - ref_y,
                    'z': pz - ref_z,
                }
            else:
                # Fixed point reference frame
                ref_x, ref_y, ref_z = reference_frame
                transformed_positions[pid] = {
                    'x': px - ref_x,
                    'y': py - ref_y,
                    'z': pz - ref_z,
                }

        # Store scatter and line artists for each particle
        scatters = []
        trails = []

        # Initialize plots for each particle
        for i, pid in enumerate(particle_ids):
            pos = transformed_positions[pid]
            # Initial scatter point
            scatter = ax.scatter3D(pos['x'][0], pos['y'][0], pos['z'][0],
                                   c=colors[i], s=sizes[i], label=f'Particle {pid}')
            scatters.append(scatter)
            # Initial trail line
            trail = ax.plot(pos['x'][0:1], pos['y'][0:1], pos['z'][0:1],
                           c=colors[i], alpha=0.5)[0]
            trails.append(trail)

        ax.legend()

        # Store references for update function
        positions_ref = transformed_positions
        pids_ref = particle_ids

        def update(frame_num):
            artists = []
            for i, pid in enumerate(pids_ref):
                pos = positions_ref[pid]

                # Determine trail start index
                if tail_length is not None:
                    trail_start = max(0, frame_num - tail_length)
                else:
                    trail_start = 0

                # Update scatter position
                scatters[i]._offsets3d = (
                    [pos['x'][frame_num]],
                    [pos['y'][frame_num]],
                    [pos['z'][frame_num]]
                )

                # Update trail
                trails[i].set_data(pos['x'][trail_start:frame_num+1],
                                   pos['y'][trail_start:frame_num+1])
                trails[i].set_3d_properties(pos['z'][trail_start:frame_num+1])

                artists.extend([scatters[i], trails[i]])

            return artists

        ani = animation.FuncAnimation(fig=fig, func=update, *args, **kwargs)
        return ani

    def plot_state_evolution(self, reference_frame=None, figsize=(12, 8), palette='tab10'):
        """
        Plots position and velocity magnitudes over time for all particles using seaborn.

        Creates a 2-row figure with position magnitude on top and velocity magnitude
        on bottom. All particles are overplotted with color coding by particle ID.

        @param reference_frame: Reference frame specification:
                               - None: Use absolute positions/velocities (no transformation).
                               - int: Compute relative to particle with this ID.
                               - 'origin' or (0,0,0): Compute relative to origin (same as absolute).
                               Default is None.
        @param figsize: Tuple (width, height) for figure size. Default (12, 8).
        @param palette: Seaborn color palette name. Default 'tab10'.
        @return: Tuple of (fig, axes) where axes is array of [ax_position, ax_velocity].
        """
        particle_ids = list(self.positions.keys())

        # Determine if we need to compute relative values
        if reference_frame == 'origin' or reference_frame == (0, 0, 0):
            relative_to = None  # Origin is same as absolute
        elif isinstance(reference_frame, int):
            relative_to = reference_frame
        else:
            relative_to = None

        # Build DataFrame for seaborn
        records = []
        for pid in particle_ids:
            pos = self.get_position(pid, relative_to=relative_to)
            vel = self.get_velocity(pid, relative_to=relative_to)

            # Compute magnitudes
            pos_mag = np.sqrt(pos['x']**2 + pos['y']**2 + pos['z']**2)
            vel_mag = np.sqrt(vel['x']**2 + vel['y']**2 + vel['z']**2)

            for i, t in enumerate(self.time):
                records.append({
                    'time': t,
                    'particle_id': pid,
                    'position': pos_mag[i],
                    'velocity': vel_mag[i],
                })

        df = pd.DataFrame(records)

        # Create figure with 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Plot position magnitude
        sns.lineplot(data=df, x='time', y='position', hue='particle_id',
                     palette=palette, ax=axes[0])
        axes[0].set_ylabel(r'$|r|$ (AU)')
        axes[0].set_title('Position Magnitude vs Time')
        axes[0].legend(title='Particle ID')

        # Plot velocity magnitude
        sns.lineplot(data=df, x='time', y='velocity', hue='particle_id',
                     palette=palette, ax=axes[1])
        axes[1].set_ylabel(r'$|v|$ (code units)')
        axes[1].set_xlabel(r'Time $(yr \cdot (2\pi)^{-1})$')
        axes[1].set_title('Velocity Magnitude vs Time')
        axes[1].legend(title='Particle ID')

        fig.tight_layout()
        return fig, axes

    def plot_state_components(self, reference_frame=None, figsize=(14, 10), palette='tab10'):
        """
        Plots position and velocity components (x, y, z) over time for all particles.

        Creates a 2x3 grid with position components on top row and velocity
        components on bottom row. All particles are overplotted with color coding.

        @param reference_frame: Reference frame specification:
                               - None: Use absolute positions/velocities.
                               - int: Compute relative to particle with this ID.
                               - 'origin' or (0,0,0): Compute relative to origin.
                               Default is None.
        @param figsize: Tuple (width, height) for figure size. Default (14, 10).
        @param palette: Seaborn color palette name. Default 'tab10'.
        @return: Tuple of (fig, axes) where axes is 2x3 array of Axes.
        """
        particle_ids = list(self.positions.keys())

        # Determine if we need to compute relative values
        if reference_frame == 'origin' or reference_frame == (0, 0, 0):
            relative_to = None
        elif isinstance(reference_frame, int):
            relative_to = reference_frame
        else:
            relative_to = None

        # Build DataFrame for seaborn
        records = []
        for pid in particle_ids:
            pos = self.get_position(pid, relative_to=relative_to)
            vel = self.get_velocity(pid, relative_to=relative_to)

            for i, t in enumerate(self.time):
                records.append({
                    'time': t,
                    'particle_id': pid,
                    'px': pos['x'][i],
                    'py': pos['y'][i],
                    'pz': pos['z'][i],
                    'vx': vel['x'][i],
                    'vy': vel['y'][i],
                    'vz': vel['z'][i],
                })

        df = pd.DataFrame(records)

        # Create figure with 2x3 subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=True)

        components = ['x', 'y', 'z']

        # Plot position components (top row)
        for col, comp in enumerate(components):
            sns.lineplot(data=df, x='time', y=f'p{comp}', hue='particle_id',
                         palette=palette, ax=axes[0, col])
            axes[0, col].set_ylabel(f'${comp}$ (AU)')
            axes[0, col].set_title(f'Position {comp.upper()}')
            axes[0, col].legend(title='Particle ID')

        # Plot velocity components (bottom row)
        for col, comp in enumerate(components):
            sns.lineplot(data=df, x='time', y=f'v{comp}', hue='particle_id',
                         palette=palette, ax=axes[1, col])
            axes[1, col].set_ylabel(f'$v_{comp}$ (code units)')
            axes[1, col].set_xlabel(r'Time $(yr \cdot (2\pi)^{-1})$')
            axes[1, col].set_title(f'Velocity {comp.upper()}')
            axes[1, col].legend(title='Particle ID')

        fig.tight_layout()
        return fig, axes


class Theorize:
    """
    Provides theoretical calculations for gravitational wave orbital decay.

    Contains static methods for computing Peters (1964) gravitational wave
    inspiral formulas, including enhancement factors and decay timescales.
    """

    def __init__(self):
        """
        Initializes a Theorize instance. Currently a placeholder for future expansion.
        """
        pass

    @staticmethod
    def enhancement_factor(e):
        """
        Computes the Peters (1964) enhancement factor f(e) for gravitational wave emission.

        The enhancement factor accounts for the increased gravitational wave
        luminosity from eccentric orbits compared to circular orbits.

        @param e: Orbital eccentricity (0 <= e < 1).
        @return: Enhancement factor f(e) as a dimensionless float.
        """
        sopra = 1 + (73/24)*e**2 + (37/96) * e**4
        sotto = (1-e**2)**(7/2)
        return sopra/sotto

    @staticmethod
    def decay_time_PN2p5(a0, m1, m2, e0):
        """
        Computes the gravitational wave inspiral time from Peters (1964).

        Calculates the time for an eccentric binary to merge due to
        gravitational wave emission at the 2.5 post-Newtonian order.

        @param a0: Initial semi-major axis (AU or astropy Quantity).
        @param m1: Mass of primary object (Msun or astropy Quantity).
        @param m2: Mass of secondary object (Msun or astropy Quantity).
        @param e0: Initial eccentricity.
        @return: Decay time as astropy Quantity in years.
        """
        a0 = ensure_unit(a0, u.AU)
        m1 = ensure_unit(m1, u.Msun)
        m2 = ensure_unit(m2, u.Msun)
        beta = (64/5) * G**3 * m1 * m2 * (m1+m2) * c**(-5)
        f = Theorize.enhancement_factor(e0)

        T = a0**4 / (4*beta*f)
        return T.to(u.yr)

    @staticmethod
    def distance(data, key, i, j):
        """
        Computes the relative distance vector between two particles or groups.

        @param data: DataFrame with columns named as key+coord+id (e.g., 'px0').
        @param key: Column prefix ('p' for position, 'v' for velocity).
        @param i: Index of first particle (int) or tuple of indices for COM.
        @param j: Index of second particle (int) or tuple of indices for COM.
        @return: Tuple (dx, dy, dz) of relative coordinate arrays.
        """
        if type(i) is int:
            xi = data[key + 'x' + str(i)]
            yi = data[key + 'y' + str(i)]
            zi = data[key + 'z' + str(i)]
        elif type(i) is tuple:
            xi, yi, zi = get_com(data, key, i)
        else:
            print('wrong index type of i')

        if type(j) is int:
            xj = data[key + 'x' + str(j)]
            yj = data[key + 'y' + str(j)]
            zj = data[key + 'z' + str(j)]
        elif type(j) is tuple:
            xj, yj, zj = get_com(data, key, j)
        else:
            print('wrong index type of j')

        return xi - xj, yi - yj, zi - zj

#-----Helper/Standalone Functions Below-----
#Calc functions

def calc_norm(x, y, z):
    """
    Computes the Euclidean norm of a 3D vector.

    @param x: X component (scalar or array).
    @param y: Y component (scalar or array).
    @param z: Z component (scalar or array).
    @return: Magnitude sqrt(x^2 + y^2 + z^2).
    """
    return np.sqrt(x ** 2 + y ** 2 + z ** 2)


def calc_ecc(m_tot, dx, dy, dz, dvx, dvy, dvz):
    """
    Computes the eccentricity vector components from state vectors.

    @param m_tot: Total mass of the system (assumes G=1).
    @param dx: Relative x position.
    @param dy: Relative y position.
    @param dz: Relative z position.
    @param dvx: Relative x velocity.
    @param dvy: Relative y velocity.
    @param dvz: Relative z velocity.
    @return: Tuple (ex, ey, ez) eccentricity vector components.
    """
    u = m_tot * 1 #G.value
    v2 = dvx ** 2 + dvy ** 2 + dvz ** 2
    r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    rv = dx * dvx + dy * dvy + dz * dvz
    ex = (dx * (v2 - u / r) - dvx * rv) / u
    ey = (dy * (v2 - u / r) - dvy * rv) / u
    ez = (dz * (v2 - u / r) - dvz * rv) / u

    return ex, ey, ez


def calc_sma(m_tot, dx, dy, dz, dvx, dvy, dvz):
    """
    Computes the semi-major axis from state vectors using the vis-viva equation.

    @param m_tot: Total mass of the system (assumes G=1).
    @param dx: Relative x position.
    @param dy: Relative y position.
    @param dz: Relative z position.
    @param dvx: Relative x velocity.
    @param dvy: Relative y velocity.
    @param dvz: Relative z velocity.
    @return: Semi-major axis a = -mu*r / (r*v^2 - 2*mu).
    """
    u = m_tot * 1 #G.value
    v2 = dvx ** 2 + dvy ** 2 + dvz ** 2
    r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return - u * r / (r * v2 - 2 * u)


def calc_angle(x1, y1, z1, x2, y2, z2):
    """
    Computes the angle between two 3D vectors.

    @param x1: X component of first vector.
    @param y1: Y component of first vector.
    @param z1: Z component of first vector.
    @param x2: X component of second vector.
    @param y2: Y component of second vector.
    @param z2: Z component of second vector.
    @return: Angle in radians between the two vectors.
    """
    r1 = calc_norm(x1, y1, z1)
    r2 = calc_norm(x2, y2, z2)
    cos = (x1 * x2 + y1 * y2 + z1 * z2) / (r1 * r2)
    return np.arccos(cos)


def calc_L(m1, m2, dx, dy, dz, dvx, dvy, dvz):
    """
    Computes the angular momentum vector components.

    @param m1: Mass of first body.
    @param m2: Mass of second body.
    @param dx: Relative x position.
    @param dy: Relative y position.
    @param dz: Relative z position.
    @param dvx: Relative x velocity.
    @param dvy: Relative y velocity.
    @param dvz: Relative z velocity.
    @return: Tuple (Lx, Ly, Lz) angular momentum components scaled by reduced mass.
    """
    m_nu = m1 * m2 / (m1 + m2)
    Lx = dy * dvz - dz * dvy
    Ly = dz * dvx - dx * dvz
    Lz = dx * dvy - dy * dvx

    return m_nu * Lx, m_nu * Ly, m_nu * Lz


def distance(data, key, i, j, npoints):
    """
    Computes relative position or velocity vectors between two particles.

    @param data: DataFrame with particle data indexed by 'id' column.
    @param key: Column prefix ('p' for position, 'v' for velocity).
    @param i: Index of first particle (int) or tuple of indices for COM.
    @param j: Index of second particle (int) or tuple of indices for COM.
    @param npoints: Number of timesteps to process.
    @return: Tuple (xdist, ydist, zdist) as lists of relative coordinates.
    """
    if type(i) is int:
        xi = data[data["id"]==i][key + 'x']
        yi = data[data["id"]==i][key + 'y']
        zi = data[data["id"]==i][key + 'z']
    elif type(i) is tuple:
        xi, yi, zi = get_com(data, key, i)
    else:
        print('wrong index type of i')

    if type(j) is int:
        xj = data[data["id"]==j][key + 'x']
        yj = data[data["id"]==j][key + 'y']
        zj = data[data["id"]==j][key + 'z']
    elif type(j) is tuple:
        xj, yj, zj = get_com(data, key, j)
    else:
        print('wrong index type of j')

    # Vectorized: use numpy array operations instead of loop
    xdist = (xi.values - xj.values)[:npoints]
    ydist = (yi.values - yj.values)[:npoints]
    zdist = (zi.values - zj.values)[:npoints]
    return xdist, ydist, zdist


def calc_h_vector(r, v):
    """
    Computes the specific angular momentum vector h = r x v.
    Based on Murray-Dermott Eq. 2.129.

    @param r: Position vector as [x, y, z] (scalars or arrays).
    @param v: Velocity vector as [vx, vy, vz] (scalars or arrays).
    @return: Tuple (hx, hy, hz) components as scalars or arrays.
    """
    x, y, z = np.asarray(r[0]), np.asarray(r[1]), np.asarray(r[2])
    vx, vy, vz = np.asarray(v[0]), np.asarray(v[1]), np.asarray(v[2])

    # Vectorized cross product components
    hx = y*vz - z*vy
    hy = z*vx - x*vz
    hz = x*vy - y*vx

    return hx, hy, hz


def get_tot_mass(data, tup):
    """
    Gets the total mass of one or more particles.

    @param data: DataFrame created by load_spacehub_data.
    @param tup: Particle index (int) or tuple of indices for multiple particles.
    @return: Total mass of the specified particle(s).
    """
    if type(tup) is int:
        return data[data["id"]==tup]["mass"][tup]
    else:
        mtot = 0

        for t in tup:
            mtot += data[data["id"]==t]["mass"][t]
        return mtot


def get_com(data, key, tup):
    """
    Computes the center of mass position or velocity for a group of particles.

    @param data: DataFrame with particle data.
    @param key: Column prefix ('p' for position, 'v' for velocity).
    @param tup: Tuple of particle indices to include in COM calculation.
    @return: Tuple (x, y, z) of mass-weighted average coordinates.
    """
    mt = get_tot_mass(data, tup)

    x = 0
    y = 0
    z = 0

    for t in tup:
        x += data[data["id"]==t]['mass'] * data[data["id"]==t][key + 'x']
        y += data[data["id"]==t]['mass'] * data[data["id"]==t][key + 'y']
        z += data[data["id"]==t]['mass'] * data[data["id"]==t][key + 'z']

    return x/mt, y/mt, z/mt


def mag(vec):
    """
    Computes the magnitude of a 3D vector or array of vectors.

    @param vec: Vector as [x, y, z] where components are scalars or arrays.
    @return: Magnitude (scalar or array).
    """
    x, y, z = np.asarray(vec[0]), np.asarray(vec[1]), np.asarray(vec[2])
    return np.sqrt(x**2 + y**2 + z**2)
#-----------Getter/Modifier functions----------
#
#------------Helpers/Intermediates---------------

def get_h_vector(data, i, j):
    """
    Computes the specific angular momentum vector from raw simulation data.

    @param data: DataFrame with particle data.
    @param i: Index of first particle (int) or tuple for COM.
    @param j: Index of second particle (int) or tuple for COM.
    @return: Tuple (hx, hy, hz) angular momentum vector components.
    """
    dx, dy, dz = distance(data, 'p', i, j)
    dvx, dvy, dvz = distance(data, 'v', i, j)
    hvec = calc_h_vector([dx, dy, dz], [dvx, dvy, dvz])

    return hvec


def add_norms(data):
    """
    Adds magnitude columns 'p' and 'v' to the DataFrame in-place.

    Computes position magnitude from (px, py, pz) and velocity magnitude
    from (vx, vy, vz).

    @param data: DataFrame to modify (modified in-place).
    """
    p_num = data["id"].nunique()
    px = data['px']
    py = data['py']
    pz = data['pz']
    data['p'] = calc_norm(px, py, pz)

    vx = data['vx']
    vy = data['vy']
    vz = data['vz']
    data['v'] = calc_norm(vx, vy, vz)


def scalar_product_xyz(vec1, vec2):
    """
    Computes the scalar (dot) product of two 3D vectors.
    Does not support lists; use for single vectors only.

    @param vec1: First vector as [x, y, z].
    @param vec2: Second vector as [x, y, z].
    @return: Scalar product vec1 . vec2.
    """
    return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]


def get_L(data, i, j):
    """
    Computes the angular momentum vector time series from simulation data.

    @param data: DataFrame with particle data.
    @param i: Index of first particle (int) or tuple for COM.
    @param j: Index of second particle (int) or tuple for COM.
    @return: Tuple (Lx, Ly, Lz) as lists of angular momentum components over time.
    """
    mi = get_tot_mass(data, i)
    mj = get_tot_mass(data, j)
    dx, dy, dz = distance(data, 'p', i, j)
    dvx, dvy, dvz = distance(data, 'v', i, j)

    Lx, Ly, Lz = [], [], []
    for t in range(0, len(dx)):
        Lxi, Lyi, Lzi = calc_L(mi, mj, dx[t], dy[t], dz[t], dvx[t], dvy[t], dvz[t])
        Lx.append(Lxi)
        Ly.append(Lyi)
        Lz.append(Lzi)

    return Lx, Ly, Lz






#----Things I wrote and don't know what to do with but they might be useful at some point----
"""def a_theory(t): from peters 5.45 for a *circular orbit*
    t = t * u.yr / (np.pi * 2)
    m1 = orb.M_i * u.Msun
    m2 = orb.M_j * u.Msun
    beta = (64/5) * G**3 * m1 * m2 * (m1+m2) * c**(-5) #Idk if the c^-5 is supposed to be here, but it fixes the units. G and c are not hard to write. stop using G=C=1
    a0 = orb.semiMajorAxis[1] * u.AU
    a_theory = (a0**4 - 4*beta*t)**(1/4)
    return a_theory.value"""



#-------------Read/Write Operations--------------

def load_spacehub_data(filename, dropna=True):
    """
    Loads SpaceHub DefaultWriter CSV output into a pandas DataFrame.

    Units in SpaceHub: AU = 1, year = 2*pi, G = 1.
    Automatically computes position and velocity magnitude columns.

    @param filename: Path to the CSV file.
    @param dropna: If True, removes rows with NaN values (default True).
    @return: DataFrame with simulation data and added norm columns.
    """
    df = pd.read_csv(filename)
    if dropna:
        df = df.dropna()
    add_norms(df)

    return df


#----Utility Functions----

def ensure_unit(x, unit: u.Unit):
    """
    Ensures a value has the correct astropy unit, converting if necessary.

    @param x: Value to check (can be scalar, Quantity, or None).
    @param unit: Desired astropy.units.Unit instance.
    @return: Value as astropy Quantity with the specified unit.
    @throws UnitTypeError: If the value cannot be converted to the target unit.
    """
    if x is None:
        return x
    if not isinstance(x, Quantity):
        x = x * unit
    elif x.unit != unit:
        try:
            x = x.to(unit)
        except u.UnitConversionError as uce:
            raise u.UnitTypeError(f"{x} cannot be converted to {unit}")
    return x


def interp_intercept(x, y, intercept=0, npoints=1e3, returnCurves=False):
    """
    Finds where an interpolated curve crosses a specified value.

    Uses PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) for
    smooth interpolation and detects zero-crossings of (y - intercept).

    @param x: Array of x-coordinates (independent variable).
    @param y: Array of y-coordinates (dependent variable).
    @param intercept: Target y-value to find crossing (default 0).
    @param npoints: Number of points for interpolation grid (default 1000).
    @param returnCurves: If True, also returns interpolated x and y arrays.
    @return: If returnCurves=False, returns [x_crossing, y_crossing].
             If returnCurves=True, returns (point, x_fine, y_fine).
             Returns [0, 0] if no crossing is found.
    """
    f = PchipInterpolator(x, y)
    x_fine = np.linspace(min(x), max(x), int(npoints))
    y_fine = f(x_fine)
    offset = y_fine - intercept
    idx_list = np.argwhere(np.diff(np.sign(offset))).flatten()
    if len(idx_list) == 0:
        print(f"No crossing found")
        return [0,0]

    idx = idx_list[0]
    point = [x_fine[idx], y_fine[idx]]
    if returnCurves:
        return point, x_fine, y_fine
    else:
        return point