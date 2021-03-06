# fft-extended-source
Public version of FFT based evaluation of extended source magnification.

FFT based method of extended soruce magnification is implemented in `magnification/mag_fft.py`. 
FFT based method of time averaged magnification is implemented in `magnification/timeave.py`.

Please cite my paper if you use my code for your project. 

FFT based method uses public FFTLog code by Xiao Fang, available at [FFTLog-and-beyond](https://github.com/xfangcosmo/FFTLog-and-beyond), and developed in [Fang et al (2019); arXiv:1911.11947](https://arxiv.org/abs/1911.11947).

`fft-extended-source` is open source and distributed with [MIT license](https://opensource.org/licenses/mit).

## Contents: notebooks and a script
- `howtouse.ipynb` shows how to use module in `magnification`. Please run `magnification/Makefile` when you want to use method developed by Lee et al. (2009).
- `timeave.ipynb` shows how to use `timeave.py` module.
- `comparison.ipynb` makes a plot of comparison of residuals by various evaluation methods of extended source magnification.
- `timeit_methods.ipynb` measures computational time of various methods.
- `paperfig.ipynb` can reproduce figures shown in the paper.
- `testdata.py` is a module to generate reference magnification using `scipy.integrate.quad`.
