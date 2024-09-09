
import argparse as ap
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as pc
from pathlib import Path

"""
https://docs.scipy.org/doc/scipy/reference/constants.html

pc.pi                               # pi = 3.141592653589793
pc.c                                # speed of light = 299792458.0 [ m s^(-1) ]
pc.e                                # charge of electron = 1.602176634e-19 [ C ]
pc.h                                # Plancck constant = 6.62607015e-34 [J s]
pc.hbar                             # Plancck constant / 2*pi = 1.0545718176461565e-34 [ J s ]
pc.centi                            # centi = 10^{-2} = 0.01
pc.hecto                            # hecto = 10^{2} = 100
pc.femto                            # femto = 10^{-15} [ s ]
pc.kilo                             # kilo = 10^{3} = 1000
pc.micron                           # micron = 10^{-6} [ m ]
pc.nano                             # nano = 10^{-9}
pc.epsilon_0                        # vacuum permittivity [ F m^−1 ]
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
                    Version:  v1.1,
                    Date:     August 11, 2024,
                    Modified: September 09, 2024""")
parser.add_argument("-t", "--type",
                    metavar="<itype_time>",
                    type=int,
                    help="The type of rt-TDDFT time for calculation, (can be 2, 22)",
                    default=22
                    )
parser.add_argument("-l", "--wavelength",
                    metavar="<wavelength>",
                    type=float,
                    help="The wavelength of laser pulse",
                    )
parser.add_argument("-p", "--power",
                    metavar="<average power>",
                    type=float,
                    help="The average power of laser pulse, in unit of mW",
                    )
parser.add_argument("-e", "--energy",
                    metavar="<laser pulse energy>",
                    type=float,
                    help="The energy of laser pulse, in unit of mJ",
                    )
parser.add_argument("-r", "--repetition",
                    metavar="<repetition rate>",
                    type=float,
                    help="The repetition rate of laser pulse, in unit of kHz",
                    default=0.001,
                    )
parser.add_argument("-D", "--diameter",
                    metavar="<diameter>",
                    type=float,
                    help="The diameter of laser pulse, in unit of micron",
                    )
parser.add_argument("-c", "--center",
                    metavar="<peak center>",
                    type=float,
                    help="The peak center of laser pulse, in unit of fs",
                    )
parser.add_argument("-w", "--fwhm",
                    metavar="<FWHM>",
                    type=float,
                    help="The Full width half maximum of laser pulse (some times we called the pulse duration)",
                    )
parser.add_argument("-dt", "--time_step",
                    metavar="<time step>",
                    type=float,
                    help="The time step of rt-TDDFT calculation, in unit of fs",
                    )
args = parser.parse_args()


def calculate_photon_energy(wavelength):
    energy = pc.h * pc.c / (wavelength * pc.nano)
    return energy / pc.e


def calculate_fluence(power, energy, time, repetition_rate, diameter):
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


def fluence_to_b1(itype, fluence, E0_in_VA, au_to_fs):
    """
    This function deals with the conversion between fluence and parameter b1.
    The unit of fluence in W/m^2, and the unit of b1 is Hartree/Bohr.

    ===========================================================================
    x = "what you want to set"
    E = x * pc.value("Hartree energy in eV") / pc.value("Bohr radius")
    I = np.sqare(E) * pc.epsilon_0 * pc.c / 2 # W/m^2
    W_per_m2_to_mW_per_cm2 = pc.kilo / pc.hecto ** 2

    J in unit of W/m^2
    ===========================================================================

    Parameters
    ----------
    itype : int
        The type of rt-TDDFT time for calculation, (can be 2, 22)
    fluence : float
        The laser fluence, in unit of W/m^2
    E0_in_VA : float
        The conversion factor of energy, (Hartree/Bohr to Volts/angstrom)
    au_to_fs : float
        The conversion factor of atomic units to fs.

    Returns
    -------
    b1 : float
        The PWmat rt-TDDFT parameter b1.
    """
    b1 = np.sqrt(2 * fluence / (pc.epsilon_0 * pc.c)) * pc.angstrom / E0_in_VA
    if itype == 2:
        return b1
    elif itype == 22:
        return b1 / au_to_fs


def get_sigma(FWHM):
    """
    This relationship between the full width half maximum (FWHM) and 
    the standard deviation (sigma) of a Gaussian function profile is:
    ==========================================
        FWHM = sigma * 2 * sqrt(2 * ln(2))    
    ==========================================

    Parameters
    ----------
    FWHM : float
        The full width half maximum of laser fluence, in unit of fs

    Returns
    -------
    sigma : float
        The sigma value of Gaussian function.

    """
    if FWHM:
        return FWHM / (2 * np.sqrt(2 * np.log(2)))
    else:
        return None


def get_time(sigma, dt):
    if sigma:
        t = 6 * sigma / 0.997
        t0 = t / 2
        return t0, np.arange(0, int(t) + dt, dt)
    else:
        return None


def get_laser_pulse(itype, E0, t0, sigma, omega, phi, t, dt):
#    y = E0 * (1/(np.sqrt(2*np.pi) * sigma)) * np.sin(omega * t + phi) * np.exp(-(t - t0) ** 2 / (2 * sigma ** 2))
    y = E0 * np.sin(omega * t + phi) * np.exp(-(t - t0) ** 2 / (2 * sigma ** 2))
    y_cum = dt * np.cumsum(y)
    if itype == 2:
        return t, y
    elif itype == 22:
        return t, y_cum


def count_non_empty_vars(b1, b2, b3, b4, b5):
    count = sum(1 for var in (b1, b2, b3, b4, b5) if var)  # count non-empty variables
    if count == 5:  # if all variables are non-empty
        return count
    else:
        return None


def save_to_file(x, y, filename='IN.TDDFT_TIME'):
    """
    """
    data = np.column_stack((x, y))
    np.savetxt(Path.cwd() / filename, data, fmt="%15.10f", delimiter="   ")


def plot_subplot(ax, time, data, label, title, ylabel):
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


def plot_figure(time, frttddft, E0_in_VA, flag):
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))

    if flag == 2:
        plot_subplot(axs[0], time, frttddft * E0_in_VA, 'E(t)', 'Field in length gauge', 'E(t) (V/Ang)')
        plot_subplot(axs[1], time, frttddft, 'E(t)', 'TDDFT_TIME in length gauge', 'TDDFT_TIME (a.u.)')
    elif flag == 22:
        plot_subplot(axs[0], time, frttddft * (-1), 'A(t)/c', 'Field in velocity gauge', 'A(t)/c (a.u.)')
        plot_subplot(axs[1], time, frttddft, 'E(t)', 'TDDFT_TIME in velocity gauge', 'TDDFT_TIME (a.u.)')

    plt.tight_layout()
    plt.savefig('TIME')
    plt.show()


def print_to_screen(wavelength, E_photon, power,fluence, energy, b1, b2, b3, b4, b5, t0, fwhm, itype, num):
    draw_dline = "=" * 79
    draw_sline = "-" * 79
    print("".join(("\n", draw_dline)))
    print(f"The energy of laser with {wavelength} nm is: {E_photon} eV.")
    if power:
        print(f"b1 = {b1}, for the input laser fluence of {fluence:.2f} W/m^2 or {fluence * (pc.kilo / (pc.hecto) ** 2):.2f} mW/cm^2.")
    elif energy:
        print(f"b1 = {b1}, for the input laser Energy density of {energy} J/m^2")
    print(f"b2 = {b2}, for the input laser peak center of {t0} fs.")
    print(f"b3 = {b3}, for the input laser wavelength of {fwhm} fs.")
    print(f"b4 = {b4}, for the input laser with FWHM of {wavelength} nm.")
    print(f"b5 = {b5}, for converting sin function to cos function.")
    print("".join(("\n", draw_sline)))
    print(f"TDDFT_TIME = {itype}, {num}, {b1}, {b2}, {b3}, {b4}, {b5}")
    print("".join((draw_dline, "\n")))


def main():
    itype = args.type                                            # 2, 22
    wavelength = args.wavelength                                 # in unit of nm
    power = args.power                                           # in unit of mW
    energy = args.energy                                         # in unit of mJ
    repetition_rate = args.repetition                            # in unit of kHz
    diameter = args.diameter                                     # in unit of micron
    center = args.center                                         # in unit of fs
    fwhm = args.fwhm                                             # in unit of fs
    dt = args.time_step                                          # in unit of fs (0.1, 1)

    # Unit conversion
    E0_in_VA = pc.value("Hartree energy in eV") / pc.value("Bohr radius") * pc.angstrom
    eV_to_au = pc.value("electron volt-hartree relationship")
    au_to_fs = pc.value("reduced Planck constant in eV s") / pc.value("Hartree energy in eV") / pc.femto

    E_photon = calculate_photon_energy(wavelength)
    sigma = get_sigma(fwhm)
    t0, t = get_time(sigma, dt)
    fluence = calculate_fluence(power, energy, t, repetition_rate, diameter)

    b1 = fluence_to_b1(itype, fluence, E0_in_VA, au_to_fs) / (1/(sigma * np.sqrt(2 * np.pi))) ## here divided the area of the Gaussian profile.
    b2 = t0
    b3 = np.sqrt(2) * sigma
    b4 = 2 * np.pi * pc.c * pc.giga * pc.femto / wavelength
    b5 = pc.pi / 2

    num = count_non_empty_vars(b1, b2, b3, b4, b5)
    if num:
        time, frttddft = get_laser_pulse(itype, b1, b2, sigma, b4, b5, t, dt)
        print_to_screen(wavelength, E_photon, power, fluence, energy, b1, b2, b3, b4, b5, t0, fwhm, itype, num)
        save_to_file(time, frttddft, filename='IN.TDDFT_TIME')
        plot_figure(time, frttddft, E0_in_VA, itype)
    else:
        print("Please check your input parameters!")


if __name__ == "__main__":
    main()
