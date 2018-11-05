

Code: 
https://github.com/GrafDuckula/CS-7641-Assignment-3

General descriptions:

This codes are for assignment 3 of Machine learning.
There are nine python files and one .sh file.

1. parse.py created all the folders for output and parse and preprocess the datasets.
2. helpers.py contains helper functions
3. plot.py contains the codes to make figures
4. ICA.py, PCA.py, RF.py, RP.py did the dimension reduction and neural network with corresponding algorithms. benchmark.py did not have any dimension reduction, it was used as benchmark for comparison.
5. clustering.py did the clustering with GMM and Kmeans.
6. ICA.py also have the codes to visualize the projection axis in new space for PCA and ICA inside.
7. clustering.py also have the codes for silhouette analysis and plot. 


To run the codes,

1. Download the datasets to the folder of python files. 
http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data
http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv

2. run parse.py to create all the output folders to store all the cvs files that will be generated and preprocess the datasets.

3. Run the benchmark.py file first, then run ICA.py, PCA.py, RF.py, RP.py to do the dimension reduction, then run the run.sh file to do the clustering, at last, run the plot.py to generate all the figures. 




Package requirements:

python 2.7
numpy == 1.15
scipy == 1.1.0
scikit-learn == 0.20
pandas == 0.22.0
matplotlib == 2.1.1
