# SNuDD

[![arXiv](http://img.shields.io/badge/arXiv-2302.12846-B31B1B.svg)](https://arxiv.org/abs/2302.12846)

**SNuDD** (**S**olar **N**e**u**trinos for **D**irect **D**etection) is a python package for accurate computations of solar neutrino scattering rates at direct detection (DD) experiments in the presence of non-standard neutrino interactions (NSI). 
**SNuDD**  was developed and utilised for the NSI sensitivity estimates of the xenon-based DD experiments XENON, LUX-ZEPLIN and DARWIN in [*A direct detection view of the neutrino NSI landscape*](https://arxiv.org/abs/2302.12846).

When using **SNuDD**, please cite:

D. W. P. Amaral, D. Cerdeno, A. Cheek and P. Foldenauer, \
*A direct detection view of the neutrino NSI landscape*,\
[arXiv:2302.12846 [hep-ph]].



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

**SNuDD** can be locally installed from the directory containing the `SNuDD` repository by running:
```bash
pip install -e SNuDD
```


## Bugs

**SNuDD** is a work in progress and so far is in an *alpha* release stage. If you find any bugs, please report them by creating an `Issue` on the project [GitHub](https://github.com/dwpamaral/SNuDD) page.
