# Quickstart

See the [notebook](https://github.com/mocquin/noise3d/tree/master/tutorials) tutorials.

# TODO : 
 - [ ] : sequences shapes : the operators of opr module return sequence of different shapes depending on the operator used, while the genseq module always return 3d T/V/H sequences --> by default all should return same sizes (3d TVH ?), and a flag could allow to switch to the other representation (where shape depends on noise type)
 - [ ] : lots of things could be done in the display module : 
     - better 3d sequence viewer : add stats, sliders
     - classic python sequence viewer
     - same for spectrum viewer
     - allow differents scale for spectrums displays : normalized freq, sample#, real-space unit
 - [ ] : masked : add numpy masked array support (for dead pixel for eg...)
 - [ ] : add a "gaussian" checker to verify the histograms are approximately normal
 - [ ] : non normal pixel exclusion : add a simple function to identify non-normal pixels, and mask them
 - [ ] : add a module to read/load real tiff sequence
 - [ ] : should define a class container for 3d sequence ?
 - [ ] : should define a class container for noise analyser ?
 - [ ] : implement all in a nice GUI