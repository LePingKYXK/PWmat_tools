
import argparse as ap
import numpy as np
import scipy.constants as pc

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
                    Version:  v1.0,
                    Date:     August 11, 2024""")
parser.add_argument("-l",
                    metavar="<wavelength>",
                    type=float,
                    help="The wavelength of laser pulse",
                    )
parser.add_argument("-p",
                    metavar="<average power>",
                    type=float,
                    help="The average power of laser pulse, in unit of mW",
                    )
parser.add_argument("-r",
                    metavar="<repeatition rate>",
                    type=float,
                    help="The repeatition rate of laser pulse, in unit of Hz",
                    )
parser.add_argument("-d",
                    metavar="<diameter>",
                    type=float,
                    help="The diameter of laser pulse, in unit of micron",
                    )
parser.add_argument("-c",
                    metavar="<peak center>",
                    type=float,
                    help="The peak center of laser pulse, in unit of fs",
                    )
parser.add_argument("-w",
                    metavar="<FWHM>",
                    type=int,
                    help="The Full width half maximum of laser pulse",
                    )
args = parser.parse_args()


def calculate_photon_energy(wavelength):
    energy = pc.h * pc.c / (wavelength * pc.nano)
    return energy / pc.e


def calculate_fluence(power, repeatition_rate, diameter):
    power_in_W = (power * pc.milli)
    rate_in_Hz = repeatition_rate * pc.kilo
    diameter_in_m = diameter * pc.micro

    fluence = power_in_W / (pc.pi * (diameter_in_m) **2 / 2)
    Edensity = power_in_W / (rate_in_Hz * pc.pi * (diameter_in_m) **2 / 2)
    return fluence, Edensity


def fluence_to_b1(fluence):
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
    fluence : float
        The laser fluence, in unit of W/m^2

    Returns
    -------
    b1 : float
        The PWmat rt-TDDFT parameter b1.

    """
    b1 = np.sqrt(2 * fluence / (pc.epsilon_0 * pc.c))
    return b1 * pc.value("Bohr radius") / pc.value("Hartree energy in eV")


def main():
    wavelength = args.l
    power = args.p
    repeatition_rate = args.r
    diameter = args.d
    center = args.c
    fwhm = args.w

    fluence, Edensity = calculate_fluence(power, repeatition_rate, diameter)
    E_photon = calculate_photon_energy(wavelength)

    b1 = fluence_to_b1(fluence)
    b2 = center
    b4 = 2 * np.pi * pc.c * pc.giga * pc.femto / wavelength
    b3 = fwhm / np.sqrt(2) / np.sqrt(np.log(4))

    draw_line = "-" * 79
    print("".join(("\n", draw_line)))
    print(f"The energy of laser with {wavelength} nm is: {E_photon} eV.")
    print(f"b1 = {b1}, for the input laser fluence of {fluence:.2f} W/m^2 or {fluence * (pc.kilo / (pc.hecto) ** 2):.2f} mW/cm^2.")
    print(f"b1 = {b1}, for the input laser Energy density of {Edensity:.2f} J/m^2")
    print(f"b2 = {b2}, for the input laser peak center of {center} fs.")
    print(f"b3 = {b3}, for the input laser wavelength of {fwhm} fs.")
    print(f"b4 = {b4}, for the input laser with FWHM of {wavelength} nm.")
    print("".join((draw_line, "\n")))


if __name__ == "__main__":
    main()
