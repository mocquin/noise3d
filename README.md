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

# Credits
Most of the calculations and concepts can be found in the following papers (sorted by publication date): 
 - [1] `John A. D'Agostino, Curtis M. Webb, "Three-dimensional analysis framework and measurement methodology for imaging system noise," Proc. SPIE 1488, Infrared Imaging Systems: Design, Analysis, Modeling, and Testing II, (1 September 1991); doi: 10.1117/12.45794` : 
 - [2] `John A. D'Agostino, "The modeling of spatial and directional noise in FLIR 90, part 1: a 3D noise analysis methodology," Proc. SPIE 2075, Passive Sensors, 20750H (29 January 1992); doi: 10.1117/12.2300245`
 - [3] `Luke B. Scott, John A. D'Agostino, "NVEOD FLIR92 thermal imaging systems performance model," Proc. SPIE 1689, Infrared Imaging Systems: Design, Analysis, Modeling, and Testing III, (16 September 1992); doi: 10.1117/12.137950`
 - [4] `James A. Dawson, Eric J. Borg, Gill L. Duykers, "Proposed standard for infrared focal-plane array nonuniformity measurements," Proc. SPIE 2224, Infrared Imaging Systems: Design, Analysis, Modeling, and Testing V, (8 July 1994); doi: 10.1117/12.180088` 
 - [5] `Curtis M. Webb, "Approach to three-dimensional noise spectral analysis," Proc. SPIE 2470, Infrared Imaging Systems: Design, Analysis, Modeling, and Testing VI, (22 May 1995); doi: 10.1117/12.210056`
 - [6] `Eddie L. Jacobs, Jae Cha, Keith A. Krapels, Van A. Hodgkin, "Assessment of 3D noise methodology for thermal sensor simulation," Proc. SPIE 4372, Infrared Imaging Systems: Design, Analysis, Modeling, and Testing XII, (10 September 2001); doi: 10.1117/12.439150`
 - [7] `Patrick O'Shea, Stephen Sousk, "Practical issues with 3D noise measurements and application to modern infrared sensors," Proc. SPIE 5784, Infrared Imaging Systems: Design, Analysis, Modeling, and Testing XVI, (12 May 2005); doi: 10.1117/12.604588`
 - [8] `Ze'ev Bomzon, "Biases in the estimation of 3D noise in thermal imagers," Proc. SPIE 7834, Electro-Optical and Infrared Systems: Technology and Applications VII, 78340B (28 October 2010); doi: 10.1117/12.864848`
 - [9] `Astrid Lundmark, "3D detector noise revisited," Proc. SPIE 8014, Infrared Imaging Systems: Design, Analysis, Modeling, and Testing XXII, 801410 (9 May 2011); doi: 10.1117/12.883259`
 - [10] `Ze'ev Bomzon, "Removing ths statistical bias from three-dimensional noise measurements," Proc. SPIE 8014, Infrared Imaging Systems: Design, Analysis, Modeling, and Testing XXII, 801416 (9 May 2011); doi: 10.1117/12.884469`
 - [11] `Finite sampling corrected 3D noise with confidence intervals, DAVID P. HAEFNER* AND STEPHEN D. BURKS, Vol. 54, No. 15 / May 20 2015 / Applied Optics, http://dx.doi.org/10.1364/AO.54.004907` 
 - [12] `Spatially Resolved 3D Noise, Infrared Imaging Systems: Design, Analysis, Modeling, and Testing XXVII, edited by Gerald C. Holst, Keith A. Krapels, Proc. of SPIE Vol. 9820, 98200K 2016 SPIE · CCC code: 0277-786X/16/18 · doi: 10.1117/12.2222905, David P. Haefner, Bradley L. Preece, Joshua M. Doe, and Stephen D. Burks, US Army RDECOM CERDEC, Night Vision & Electronic Sensors Directorate, 10221 Burbeck Road, Fort Belvoir, VA, USA`
 - [13] `David P. Haefner, "Power spectral density of 3D noise," Proc. SPIE 10178, Infrared Imaging Systems: Design, Analysis, Modeling, and Testing XXVIII, 101780D (3 May 2017); doi: 10.1117/12.2260885`

