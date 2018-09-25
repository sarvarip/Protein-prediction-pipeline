Option 1: just run t-sne\
Simply copy the csv files into your cwd and run the jupyter notebook code to produce the plots\
Option 2: advanced, the whole process\
a) Use tsne_modelbuilder.py to change model\
b) Run tsne_antibac_aac.py to generate the machine learning model and to have it saved on the disk\
c) Run tsne_antibac_main.py to get the features in the last layer of the neural network\
(lines 96-102 need to be changed if you changed the model in step a)) and to have them extracted in cvs files\
d) Run the jupyter notebook file (alternatively use tsne_calc_plot.py)\

