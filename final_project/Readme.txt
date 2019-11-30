A Network Embedding Model with the Variants of Autoencoder 


CODE FILES INCLUDE:
FinalModel.m           ---- the main file that implement the proposed network embedding model
classificationACC.m    ---- file used to calculate the accuracy of classification
ClusteringMeasure.m    ---- file used to calculate the accuarcy, NMI and purity of clustering

DATASETS INCLUDE:
cornell.mat            ---- small dataset
texas.mat              ---- small dataset
washington.mat         ---- small dataset
wisconsin.mat          ---- small dataset
BlogCatalog.mat        ---- large dataset
Flickr.mat             ---- large dataset

The properties of the datasets are summarized below.
Dataset      #Nodes	 #Edges	 #Attributes  #Labels
cornell      195     286     1703         5
texas        195     298     1703         5
washington   195     417     1703         5
wisconsin    195     479     1703         5
BlogCatalog  5196    171743	 8189         6
Flickr       7575    239738	 12047        9




INSTALLATION AND REQUIREMENT
It is required to install the LIBSVM(https://www.csie.ntu.edu.tw/~cjlin/libsvm/) package in matlab.
The codes have been tested in Matlab R2016b 64bit edition.

