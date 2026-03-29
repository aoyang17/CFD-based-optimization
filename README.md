# CRM Wing CFD Optimization

This repository contains scripts for **CFD-based optimization of the CRM wing**, including both **FFD-based** and **mode-based** parameterization approaches.

## Folder Overview

### `/CFD-FFD-Optimization/`

This folder contains scripts for **FFD-based CFD optimization** of the CRM wing using **MACH-Aero** with an **FFD box**.

It includes:

- **Single-point optimization**: ADODG Case 4.1
- **Nine-point optimization**: ADODG Case 4.5

For the detailed problem settings and benchmarks, please refer to the ADODG website:  
<https://sites.google.com/view/mcgill-computational-aerogroup/adodg>

### `/CFD-Mode-Optimization/`

This folder contains scripts for the **mode-based parameterization approach** developed by **Jichao Li**. Please cite the related references if you use this approach in your work.

Although some modules are adapted from the `pyGeo` framework for convenient coupling with pyGeo-based workflows, this method is **not fundamentally based on FFD**.

In particular:

- `DVGeometry_FFD_MODE.py` is intended for compatibility with **older versions of pyGeo**
- `DVGeometry_MODE_update.py` is intended for compatibility with **newer versions of pyGeo**

Both versions are usable.

## Notes

- The **FFD-based approach** and the **mode-based approach** are separate parameterization strategies.
- The mode-based implementation only reuses certain `pyGeo` modules for integration convenience and should not be interpreted as an FFD method.

## Learning Resources and References

### MACH-Aero / FFD-based Optimization

- [MACH-Aero Framework Overview](https://mdolab-mach-aero.readthedocs-hosted.com/en/latest/machFramework/MACH-Aero.html)  
  Official overview of the MACH-Aero framework, including pyGeo, IDWarp, ADflow, and optimization workflow.

- [Wing Aerodynamic Optimization Tutorial](https://mdolab-mach-aero.readthedocs-hosted.com/en/latest/machAeroTutorials/opt_overview.html)  
  A practical starting point for CRM-style wing optimization in MACH-Aero.

- [Aerodynamic Optimization Tutorial](https://mdolab-mach-aero.readthedocs-hosted.com/en/latest/machAeroTutorials/opt_aero.html)  
  Official tutorial showing how to set up aerodynamic shape optimization using MACH-Aero.

- [Geometric Parameterization with FFD in pyGeo](https://mdolab-mach-aero.readthedocs-hosted.com/en/latest/machAeroTutorials/opt_ffd.html)  
  Official tutorial on FFD-based geometric parameterization, including reference axes and design variables.

- Martins, J. R. R. A., and Ning, A., *Engineering Design Optimization*. Cambridge University Press, 2021.  
  DOI: [10.1017/9781108980647](https://doi.org/10.1017/9781108980647)

### Mode-based Parameterization (Jichao Li)

- Li, J., and Zhang, M., “Adjoint-Free Aerodynamic Shape Optimization of the Common Research Model Wing,” *AIAA Journal*, Vol. 59, No. 6, 2021, pp. 1990–2000.  
  DOI: [10.2514/1.J059921](https://doi.org/10.2514/1.J059921)

- Li, J., and Zhang, M., “Data-Based Approach for Wing Shape Design Optimization,” *Aerospace Science and Technology*, Vol. 112, 2021, 106639.  
  DOI: [10.1016/j.ast.2021.106639](https://doi.org/10.1016/j.ast.2021.106639)

- Li, J., and Zhang, M., “On Deep-Learning-Based Geometric Filtering in Aerodynamic Shape Optimization,” *Aerospace Science and Technology*, Vol. 112, 2021, 106603.  
  DOI: [10.1016/j.ast.2021.106603](https://doi.org/10.1016/j.ast.2021.106603)
