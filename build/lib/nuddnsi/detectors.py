"""Detectors are bundled together here."""

from dataclasses import dataclass

from nuddnsi.bkg import lz_background, xnt_background, dw_background
from nuddnsi.efficiencies import Efficiency, efficiency_lz_nr, efficiency_lz_er, efficiency_xnt_nr, efficiency_xnt_er
from nuddnsi.resolution import Resolution, res_lz_nr, res_lz_er, res_xnt_nr, res_xnt_er
from nuddnsi.targets import Nucleus, Electron, nucleus_xe, electron_xe


@dataclass
class Detector:
    """Bundles together detector-specific specs (right now just a bundle; later will have functionality)."""

    nucleus: Nucleus  # TODO: Bundle these two together
    electron: Electron
    exposure: float
    efficiency_nr: Efficiency  # If you want the log-extended efficiency here, feed that in instead (TODO: Make neater)
    efficiency_er: Efficiency
    resolution_nr: Resolution
    resolution_er: Resolution  # TODO: Think about adding Convolver here
    electron_bkg: callable


lz = Detector(nucleus_xe,
              electron_xe,
              15.34,
              efficiency_lz_nr,
              efficiency_lz_er,
              res_lz_nr,
              res_lz_er,
              lz_background.total_bkg_spec_fn)

xnt = Detector(nucleus_xe,
               electron_xe,
               20.,
               efficiency_xnt_nr,
               efficiency_xnt_er,
               res_xnt_nr,
               res_xnt_er,
               xnt_background.total_bkg_spec_fn)

darwin = Detector(nucleus_xe,
                  electron_xe,
                  200.,
                  efficiency_xnt_nr,
                  efficiency_xnt_er,
                  res_xnt_nr,
                  res_xnt_er,
                  dw_background.total_bkg_spec_fn)


def main():
    """Run a quick example of how to access attributes (using DARWIN). You get things by accessing the attributes!"""
    exp = darwin.exposure
    efficiency_nr = darwin.efficiency_nr  # You could use efficiency and resolution for the convolution
    res_nr = darwin.resolution_nr
    E_thresh = efficiency_nr.threshold_50  # The 50% threshold is inside the efficiency function!! (TODO: Bad.)

    print(f"Darwin has an exposure of {exp} ton-yr, a threshold of {E_thresh * 1e6} keV_nr, an efficiency function "
          f"{efficiency_nr}, and a resolution function {res_nr}.")


if __name__ == "__main__":
    main()
