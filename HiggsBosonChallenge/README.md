Higgs Boson Machine Learning Challenge
======

<img src="https://kaggle2.blob.core.windows.net/competitions/kaggle/3887/media/ATLASEXP_image.png" alt="ATLAS" title="ATLAS" />

<a href="http://www.kaggle.com/c/higgs-boson">View the competition details here.</a><br/>

This directory includes the code I used to run experiments for the competition.  Despite starting only a few weeks before the deadline and having very limited time to invest, I managed to place in the top 25%.<br/>

I used the Anaconda distribution of Python with the IPython kernel and Spyder IDE to run experiments.  I also installed and configured several additional dependencies (xgboost and pylearn 2).  There are three scripts of interest:

higgs.py - Primarity based on scikit-learn solutions
higgs-adv.py - Switched to a Linux VM and incorporated the xgboost library
higgs-nn.py - Set up pylearn 2 and started experimenting with deep learning neural nets (unfortunately I ran out of time before making any significant progress here)

The scripts are fairly basic due to time constraints but well-modularized and easy to follow.  Enjoy!
