# SNuDD

**SNuDD** (**S**olar **N**e**u**trinos for **D**irect **D**etection) is a python package for accurate computations of solar neutrino scattering rates at direct detection (DD) experiments in the presence of non-standard neutrino interactions (NSI). 
**SNuDD**  was developed and utilised for the NSI sensitivity estimates of the xenon-based DD experiments XENON, LUX-ZEPLIN and DARWIN in [*A direct detection view of the neutrino NSI landscape*](https://arxiv.org/abs/2302.12846).


## Prerequisites

**SNuDD** does not have any external dependencies. It relies, however, on the python modules 

- `numpy`: v>= 1.20.3
- `matplotlib`: v>=3.3.4
- `scipy`: v>=1.6.2
- `pandas`: v>=1.3.0
- `setuptools`: v>=52.0.0
- `pymultinest`: v>=2.11



## Installation

You can obtain the sources directly from the [github repository](https://github.com/dwpamaral/SNuDD) either by using `git`:
```bash
git clone https://github.com/dwpamaral/SNuDD.git
```
or by downloading them as a tarball:
```bash
wget https://github.com/dwpamaral/SNuDD/archive/master.tar.gz
tar -xzvf master.tar.gz
mv SNuDD-main SNuDD
```

**SNuDD** can then be locally installed by running:
```bash
pip install -e snudd
```
