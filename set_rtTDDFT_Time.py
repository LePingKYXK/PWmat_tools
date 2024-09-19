
import argparse as ap
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as pc
from pathlib import Path
from scipy.integrate import simps
from scipy.optimize import minimize

"""
https://docs.scipy.org/doc/scipy/reference/constants.html

pc.pi                               # pi = 3.141592653589793
pc.c                                # speed of light = 299792458.0 [ m s^{-1} ]
pc.e                                # charge of electron = 1.602176634e-19 [ C ]
pc.h                                # Plancck constant = 6.62607015e-34 [J s]
pc.hbar                             # Plancck constant / 2*pi = 1.0545718176461565e-34 [ J s ]
pc.centi                            # centi = 10^{-2} = 0.01
pc.hecto                            # hecto = 10^{2} = 100
pc.femto                            # femto = 10^{-15} [ s ]
pc.kilo                             # kilo = 10^{3} = 1000
pc.micron                           # micron = 10^{-6} [ m ]
pc.nano                             # nano = 10^{-9}
pc.epsilon_0                        # vacuum permittivity [ F m^{−1} ]
pc.value("Bohr radius")             # Bohr radius [ m ]
pc.value("Hartree energy")          # Hartree Energy [ J ]
pc.value("Hartree energy in eV")    # Hartree energy [ eV ]
pc.angstrom                         # 1E-10 [ m ]

"""


parser = ap.ArgumentParser(add_help=True,
                    formatter_class=ap.ArgumentDefaultsHelpFormatter,
                    description="""
                    Author:   Dr. Huan Wang,
                    Email:    huan.wang@whut.edu.cn,
                    Version:  v1.2,
                    Date:     August 11, 2024,
                    Modified: September 19, 2024""")
parser.add_argument("-t", "--type",
                    metavar="<itype_time>",
                    type=int,
                    help="The type of rt-TDDFT time for calculation, (can be 2, 22)",
                    default=22
                    )
parser.add_argument("-l", "--wavelength",
                    metavar="<wavelength>",
                    type=float,
                    help="The wavelength of the laser pulse, in the unit of nm",
                    )
parser.add_argument("-p", "--power",
                    metavar="<average power>",
                    type=float,
                    help="The average power of laser pulse, in the unit of mW",
                    )
parser.add_argument("-e", "--energy",
                    metavar="<laser pulse energy>",
                    type=float,
                    help="The energy of laser pulse, in the unit of mJ",
                    )
parser.add_argument("-r", "--repetition",
                    metavar="<repetition rate>",
                    type=float,
                    help="The repetition rate of the laser pulse, in the unit of kHz",
                    default=0.001,
                    )
parser.add_argument("-D", "--diameter",
                    metavar="<diameter>",
                    type=float,
                    help="The diameter of the laser pulse, in the unit of micron",
                    )
parser.add_argument("-c", "--center",
                    metavar="<peak center>",
                    type=float,
                    help="The peak center of the laser pulse, in the unit of fs",
                    )
parser.add_argument("-w", "--fwhm",
                    metavar="<FWHM>",
                    type=float,
                    help="The Full width half maximum of the laser pulse (sometimes we call the pulse duration), in the unit of fs",
                    )
parser.add_argument("-dt", "--time_step",
                    metavar="<time step>",
                    type=float,
                    help="The time step of rt-TDDFT calculation, in the unit of fs",
                    )
args = parser.parse_args()


def calculate_photon_energy(wavelength):
    """
    This function calculates the photon energy from the wavelength.

    Parameters
    ----------
    wavelength : float
        The wavelength of the laser pulse, in the unit of nm.

    Returns
    -------
    energy : float
        The photon energy, in the unit of J.
    """
    energy = pc.h * pc.c / (wavelength * pc.nano)
    return energy / pc.e


def calculate_fluence(power, energy, time, repetition_rate, diameter):
    """
    This function calculates the fluence of the laser pulse.

    Parameters
    ----------
    power : float
        The average power of laser pulse, in the unit of mW.
    energy : float
        The energy of the laser pulse, in the unit of mJ.
    time : np.array
        The 1d-array contains the time of laser pulse, in the unit of fs.
    repetition_rate : float
        The repetition rate of laser pulse, in the unit of kHz.
    diameter : float
        The diameter of the laser pulse, in the unit of micron.

    Returns
    -------
    fluence : np.array
        The 1d-array contains the fluence of the laser pulse, in the unit of W/m^2
    """
    diameter_in_m = diameter * pc.micro
    rate_in_Hz = repetition_rate * pc.kilo
    if power:
        power_in_W = (power * pc.milli)
        fluence_1photon = power_in_W / (pc.pi * (diameter_in_m / 2) **2)
        return fluence_1photon

    elif energy:
        energy_in_J = (energy * pc.milli) 
        power_in_W = energy_in_J * time
        fluence_average = power_in_W / (pc.pi * (diameter_in_m / 2) **2) * rate_in_Hz
        return fluence_average


def unit_conversion_fluence(itype: int, fluence: float, E0_in_VA: float, au_to_fs: float) -> float:
    """
    This function deals with the unit conversion of fluence.
    The unit of fluence is W/m^2, and the unit of E0 is Hartree/Bohr.

    ===========================================================================
    x = "what you want to set"
    E = x * pc.value("Hartree energy in eV") / pc.value("Bohr radius")
    I = np.sqare(E) * pc.epsilon_0 * pc.c / 2 # W/m^2
    W_per_m2_to_mW_per_cm2 = pc.kilo / pc.hecto ** 2

    fluence in unit of W/m^2
    ===========================================================================

    Parameters
    ----------
    itype : int
        The type of rt-TDDFT time for calculation, (can be 2, 22)
    fluence : float
        The laser fluence, in the unit of W/m^2
    E0_in_VA : float
        The conversion factor of energy, (Hartree/Bohr to Volts/angstrom)
    au_to_fs : float
        The conversion factor of atomic units to fs.

    Returns
    -------
    E0 : float
        The PWmat rt-TDDFT parameter b1.
    """
    E0 = np.sqrt(2 * fluence / (pc.epsilon_0 * pc.c)) * pc.angstrom / E0_in_VA
    if itype == 2:
        return E0
    elif itype == 22:
        return E0 / au_to_fs


def calculate_sigma(FWHM):
    """
    This relationship between the full width half maximum (FWHM) and 
    the standard deviation (sigma) of a Gaussian function profile is:
    ==========================================
        FWHM = sigma * 2 * sqrt(2 * ln(2))    
    ==========================================

    Parameters
    ----------
    FWHM : float
        The full width half maximum of laser fluence, in the unit of fs

    Returns
    -------
    sigma : float
        The sigma value of Gaussian function.

    """
    if FWHM:
        return FWHM / (2 * np.sqrt(2 * np.log(2)))
    else:
        return None


def calculate_time(sigma, dt):
    """
    This function calculates the sigma and time array for the laser pulse.

    Parameters
    ----------
    sigma : float
        The standard deviation of Gaussian function.
    dt : float
        The time step of rt-TDDFT calculation, in unit of fs

    Returns
    -------
    t0 : float
        The time of laser pulse center, in unit of fs.
    t : np.array
        The 1d-array contains the time of laser pulse, in unit of fs.
    """
    if sigma:
        t = 6 * sigma / 0.997
        t0 = t / 2
        return t0, np.arange(0, int(t) + dt, dt)
    else:
        return None


def integrand(t: np.array, b1: float, b2: float, b3: float, b4: float, b5: float):
    """
    This function generates the integrand of the laser pulse shape, which is 
    a product of a Gaussian function multiplied by a sine function.
    
    Parameters
    ----------
    t : np.array
        The 1d-array contains the time of the laser pulse, in the unit of fs.
    b1 : float
        The PWmat rt-TDDFT parameter b1.
    b2 : float
        The PWmat rt-TDDFT parameter b2.
    b3 : float
        The PWmat rt-TDDFT parameter b3.
    b4 : float
        The PWmat rt-TDDFT parameter b4.
    b5 : float
        The PWmat rt-TDDFT parameter b5.

    Returns
    -------
    integrand : np.array
        The 1d-array contains the integrand of the laser pulse shape.

    """    
    gaussian = b1 * np.exp(-((t - b2)**2) / (2 * b3**2))
    sine = np.sin(b4 * t + b5)
    return gaussian * sine


def loss_function(b1: float, b2: float, b3: float, b4: float, b5: float, time_array: np.array, dt: float, desired_integral_value: float) -> float:
    """
    This function calculates the loss function for the laser pulse shape.
    The loss function is the absolute difference between the integral of the laser pulse shape and the desired integral value.
    
    Parameters
    ----------
    b1 : float
        The PWmat rt-TDDFT parameter b1.
    b2 : float
        The PWmat rt-TDDFT parameter b2.
    b3 : float
        The PWmat rt-TDDFT parameter b3.
    b4 : float
        The PWmat rt-TDDFT parameter b4.
    b5 : float
        The PWmat rt-TDDFT parameter b5.
    time_array : np.array
        The 1d-array contains the time series of the laser pulse, in the unit of fs.
    dt : float
        The time step of rt-TDDFT calculation, in the unit of fs.
    desired_integral_value : float
        The desired integral value of the laser pulse shape.

    Returns
    -------
    loss : float
        The loss value of the laser pulse shape.

    """
    integrand_values = integrand(time_array, b1, b2, b3, b4, b5)
    integral_value = simps(integrand_values, time_array)
    return np.abs(integral_value - desired_integral_value)


def generate_laser_pulse(itype: int, E0: float, t0: float, sigma: float, omega: float, phi: float, t: np.array, dt: float) -> np.array:
    """
    This function calculates the laser pulse shape.

    Parameters
    ----------
    itype : int
        The type of rt-TDDFT time for calculation, (can be 2, 22)
    E0 : float
        The PWmat rt-TDDFT parameter b1.
    t0 : float
        The time of laser pulse center, in the unit of fs.
    sigma : float
        The standard deviation of the Gaussian function.
    omega : float
        The angular frequency of laser pulse, in the unit of rad/fs.
    phi : float
        The phase of the laser pulse, in the unit of rad.
    t : np.array
        The 1d-array contains the time of laser pulse, in the unit of fs.
    dt : float
        The time step of rt-TDDFT calculation, in the unit of fs.
    
    Returns
    -------
    t : np.array
        The 1d-array contains the time of laser pulse, in the unit of fs.
    f_rttddft : np.array
        The 1d-array contains the laser pulse shape, in the unit of V/angstrom or a.u.

    """
    f_rttddft = E0 * np.sin(omega * t + phi) * np.exp(-(t - t0) ** 2 / (2 * sigma ** 2))
    f_rttddft_cum = dt * np.cumsum(y)
    if itype == 2:
        return t, f_rttddft
    elif itype == 22:
        return t, f_rttddft_cum


def count_non_empty_vars(b1, b2, b3, b4, b5):
    """
    This function counts the number of non-empty variables.

    Parameters
    ----------
    b1 : float
        The PWmat rt-TDDFT parameter b1.
    b2 : float
        The PWmat rt-TDDFT parameter b2.
    b3 : float
        The PWmat rt-TDDFT parameter b3.
    b4 : float
        The PWmat rt-TDDFT parameter b4.
    b5 : float
        The PWmat rt-TDDFT parameter b5.

    Returns
    -------
    count : int
        The number of non-empty variables.
    """
    count = sum(1 for var in (b1, b2, b3, b4, b5) if var)  # count non-empty variables
    if count == 5:  # if all variables are non-empty
        return count
    else:
        return None


def save_to_file(x, y, filename='IN.TDDFT_TIME'):
    """
    This function saves the data to the "IN.TDDFT_TIME" file.

    Parameters
    ----------
    x : np.array
        The 1d-array contains the time of laser pulse, in the unit of fs.
    y : np.array
        The 1d-array contains the laser pulse shape, in the unit of V/angstrom or a.u.
    filename : str, optional
        The name of the file to save the data. The default is 'IN.TDDFT_TIME'.

    Returns
    -------
    None.
    """
    data = np.column_stack((x, y))
    np.savetxt(Path.cwd() / filename, data, fmt="%15.10f", delimiter="   ")


def plot_subplot(ax, time, data, label, title, ylabel):
    """
    This function supports the plot_figure function to plot the data in a subplot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The subplot to plot the data.
    time : np.array
        The 1d-array contains the time of laser pulse, in the unit of fs.
    data : np.array
        The 1d-array contains the laser pulse shape, in the unit of V/angstrom or a.u.
    label : str
        The label of the data.
    title : str
        The title of the subplot.
    ylabel : str
        The Y-axis label of the subplot.

    Returns
    -------
    None.
    """
    ax.plot(time, data, label=label)
    ax.set_title(title)
    ax.set_xlim(0, time.max())
    x_ticks = list(ax.get_xticks()) + [round(time.max(), 2)]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)


def plot_figure(time, f_rttddft, E0_in_VA, flag):
    """
    This function plots the laser pulse shape and the TDDFT_TIME.

    Parameters
    ----------
    time : np.array
        The 1d-array contains the time of laser pulse, in the unit of fs.
    f_rttddft : np.array
        The 1d-array contains the laser pulse shape, in the unit of V/angstrom or a.u.
    E0_in_VA : float
        The conversion factor of energy, (Hartree/Bohr to Volts/angstrom)
    flag : int
        The type of rt-TDDFT time for calculation, (can be 2, 22)
    
    Returns
    -------
    None.
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))

    if flag == 2:
        plot_subplot(axs[0], time, f_rttddft * E0_in_VA, 'E(t)', 'Field in length gauge', 'E(t) (V/Ang)')
        plot_subplot(axs[1], time, f_rttddft, 'E(t)', 'TDDFT_TIME in length gauge', 'TDDFT_TIME (a.u.)')
    elif flag == 22:
        plot_subplot(axs[0], time, f_rttddft * (-1), 'A(t)/c', 'Field in velocity gauge', 'A(t)/c (a.u.)')
        plot_subplot(axs[1], time, f_rttddft, 'E(t)', 'TDDFT_TIME in velocity gauge', 'TDDFT_TIME (a.u.)')

    plt.tight_layout()
    plt.savefig('TIME')
    plt.show()


def print_to_screen(wavelength, E_photon, power, fluence, energy, b1, b2, b3, b4, b5, t0, dt, fwhm, itype, num):
    draw_dline = "=" * 79
    draw_sline = "-" * 79
    draw_exclamation = "!" * 79
    print("".join(("\n", draw_dline)))
    print(f"The energy of laser with {wavelength} nm is {E_photon:.2f} eV.")

    if 1 / dt < 2 * b4:    # The Nyquist sampling frequency
        print("".join(("\n", draw_exclamation)))
        print(f"The omega of laser with {wavelength} nm is {b4:.2f} rad/fs.")
        print(f"Warning: The time step {dt} is too close to the value of omega {b4}.\nPlease DECREASE the time step, not greater than {1 /(2 * b4):.2f} fs.")
        print("".join((draw_exclamation, "\n")))

    if power:
        print(f"b1 = {b1}, for the input laser fluence of {fluence:.2f} W/m^2 or {fluence * (pc.kilo / (pc.hecto) ** 2):.2f} mW/cm^2.")
    elif energy:
        print(f"b1 = {b1}, for the input laser Energy density of {energy:.2f} J/m^2")

    print(f"b2 = {b2}, for the input laser peak center at {t0:.2f} fs.")
    print(f"b3 = {b3}, for the input laser with the FWHM of {fwhm} fs.")
    print(f"b4 = {b4}, for the input laser wavelength of {wavelength} nm.")
    print(f"b5 = {b5}, for converting sin function to cos function.")
    print("".join(("\n", draw_sline)))
    print(f"TDDFT_TIME = {itype}, {num}, {b1}, {b2}, {b3}, {b4}, {b5}")
    print("".join((draw_dline, "\n")))


def main():
    itype = args.type                                            # 2 or 22
    wavelength = args.wavelength                                 # in the unit of nm
    power = args.power                                           # in the unit of mW
    energy = args.energy                                         # in the unit of mJ
    repetition_rate = args.repetition                            # in the unit of kHz
    diameter = args.diameter                                     # in the unit of micron
    center = args.center                                         # in the unit of fs
    fwhm = args.fwhm                                             # in the unit of fs
    dt = args.time_step                                          # in the unit of fs (0.1, 1)

    # Unit conversion
    E0_in_VA = pc.value("Hartree energy in eV") / pc.value("Bohr radius") * pc.angstrom
    eV_to_au = pc.value("electron volt-hartree relationship")
    au_to_fs = pc.value("reduced Planck constant in eV s") / pc.value("Hartree energy in eV") / pc.femto

    E_photon = calculate_photon_energy(wavelength)
    sigma = calculate_sigma(fwhm)
    t0, time_array = calculate_time(sigma, dt)
    fluence = calculate_fluence(power, energy, time_array, repetition_rate, diameter)
    flu = unit_conversion_fluence(itype, fluence, E0_in_VA, au_to_fs)

    initial_guess = [1.0]
    b2 = t0
    b3 = np.sqrt(2) * sigma
    b4 = 2 * np.pi * pc.c * pc.giga * pc.femto / wavelength
    b5 = pc.pi / 2
    results = minimize(loss_function, initial_guess, args=(b2, b3, b4, b5, time_array, dt, flu))
    b1 = np.abs(results.x[0])

    num = count_non_empty_vars(b1, b2, b3, b4, b5)
    if num:
        time, f_rttddft = generate_laser_pulse(itype, b1, b2, sigma, b4, b5, time_array, dt)
        print_to_screen(wavelength, E_photon, power, fluence, energy, b1, b2, b3, b4, b5, t0, dt, fwhm, itype, num)
        save_to_file(time, f_rttddft, filename='IN.TDDFT_TIME')
        plot_figure(time, f_rttddft, E0_in_VA, itype)
    else:
        print("Please check your input parameters!")


if __name__ == "__main__":
    main()
