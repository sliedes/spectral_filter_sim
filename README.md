# A Jupyter notebook to simulate spectral filters

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sliedes/spectral_filter_sim/HEAD?labpath=Spectral%20filter%20simulator.ipynb)

## To run

Just [Launch Binder](https://mybinder.org/v2/gh/sliedes/spectral_filter_sim/HEAD?labpath=Spectral%20filter%20simulator.ipynb) and then click the ⏩ button or select *Menu->Run->Restart kernel and run all cells...* to try it out. Note that it may take a while to load even before you get to see that button and menu. Then scroll down to see the image and the controls.

## What is it?

You can use this notebook to approximate the effects of viewing scenes in the [CAVE Multispectral Image Database](https://www.cs.columbia.edu/CAVE/databases/multispectral/) in daylight through a spectral filter.
An example of such a spectral filter could be glasses that have a notch to enhance separation between red and green colors, probably somewhere around the 550-600 nm range.

However, note that this is complicated by the fact that your monitor in fact only displays red, green and blue light, so the spectrum emitted by it will not be the same thing.

Essentially, this is a crude approximation. The one (rather useless) thing it should work extremely well for is showing what the world would look like when viewed through such a filter—for a person with normal color vision. I assume that the results should be indicative for most common types of anomalous trichromacy, including the most common form of red-green color blindness (deuteranomalia) where the spectral response of the M cone cell has shifted.
