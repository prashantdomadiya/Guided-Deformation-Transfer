# Guided-Deformation-Transfer
Deformation Transfer method for various geometric models

# OS
I tested Add-On on Ubuntu. It may work on Windows (with few changes in code).

# Blender Version
  2.81 or more

# Dependencies
Numpy, Scipy, scikit-sparse

# Blender set up through Anaconda (if unable to install dierectly on system)
1. Install Anaconda
2. Create an python environment name "Blender281" (version of the python should be the same as Blender python)
3. install numpy, scipy and sciki-sparse
4. uninstall numpy and scipy from blender (otherwise you will face comaptibility issue)
6. Anaconda folder-> envs -> Blender281 -> lib -> PythonX.Y -> site-packages
5. replace global variable "LibPath=Anaconda3/Blender281/lib/Python3.7/site-packages/" by 
   "LibPath=Your_Anaconda_folder_name/Blender281/lib/PythonX.Y/site-packages/"
6. Activate the Add-On.
   The toolbox is visible in blender UI region.
   
    

# How to use Deformation Transfer Toolbox
please see the video at https://youtu.be/1kJnbTPwzuI
# Any Query contact me on
pmdomadiya@gmail.com

# Please cite following paper

@inproceedings{Domadiya:2019:GDT:3359998.3369408,
 author = {Domadiya, Prashant M and Shah, Dr.Pratik and Mitra, Suman},
 title = {Guided Deformation Transfer},
 booktitle = {European Conference on Visual Media Production},
 series = {CVMP '19},
 year = {2019},
 isbn = {978-1-4503-7003-5},
 location = {London, United Kingdom},
 pages = {7:1--7:10},
 articleno = {7},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/3359998.3369408},
 doi = {10.1145/3359998.3369408},
 acmid = {3369408},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {Deformation Transfer, Poisson Interpolation, Vector Graph},
} 
