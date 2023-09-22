##BEFORE running any scripts, remember to conda-activate the MicNet and then ipython

##import required libraries

import hdbscan
import micnet as mc #after conda installation, this is pip-installed
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import show
from micnet.utils import filter_otus

##define root directory
ROOT_DIR="/Users/ashley/Documents/Research/FGL_chapters_to_manuscripts/traps/"

##import data

#import amplicon count data as a pandas dataframe where ASV identifiers are indices and
#individual samples are columns. optional first column=taxonomy. I am choosing to omit taxnomy.

#biom conversion process leaves one line of text before data. Skip over. TSV so specify delimiting character
ampData = pd.read_csv(ROOT_DIR+"qiimemergeoutputs/exported_FGL_wc_traps_merged_table_filt_w_chloromito/exp_table_filtwchloromitoed.tsv",skiprows=[0],delimiter="\t")

##set metrics/parameters for HDBSCAN and UMAP

#metrics for umap and hdbscan can be picked from the following:
METRIC_UMAP=["euclidean","manhattan","canberra","braycurtis", "cosine","correlation","hellinger"]
METRIC_HDB=["euclidean","manhattan","canberra","braycurtis"]

#parameters
n_neighbors = 2
min_dist = 1
n_components = 2
metric_umap = METRIC_UMAP[3]
metric_hdb = METRIC_HDB[3]
min_cluster_size = 2
min_sample = 3

#create the dimension reduction class- I have chosen to use Bray-Curtis method as opposed to kombucha test data set (euclidean)
embedding_outliers=mc.Embedding_Output(n_neighbors=n_neighbors,min_dist=min_dist, n_components=n_components,
                                    metric_umap=metric_umap,metric_hdb=metric_hdb,min_cluster_size=min_cluster_size,
                                    min_sample=min_sample,output=True)


#instatiate class SparCC_MicNet with desired parameter values of the parameters for co-occurence network

#first, SparCC is run without any ASV or taxa on unprocessed original data
dataSparcc = ampData.iloc[:,1:]
print(dataSparcc.shape)
dataSparcc.head()

#second, set parameters for SparCC- i will change these later for FGL dataset
n_iteractions=3
x_iteractions=3
low_abundance=True
threshold=0.1
normalization="dirichlet"
log_transform=True
num_simulate_data=5
type_pvalues="one_sided"

#Create Sparcc object
SparCC_MN = mc.SparCC_MicNet(n_iteractions=n_iteractions,
                                    x_iteractions=x_iteractions,
                                    low_abundance=low_abundance,
                                    threshold=threshold,
                                    normalization=normalization,
                                    log_transform=log_transform,
                                    num_simulate_data=num_simulate_data,
                                    type_pvalues=type_pvalues,
                                    )

##Body

#quick check
print(ampData.shape)
ampData.head()

#pre-process data
#filtering functions require you to specify if data has taxa information

#singleton filtration removes ASVs that have less than a total of five occurences among all samples
singletonFiltAmpData,ASV_ids=filter_otus(ampData,taxa=False,low_abundance=False) #setting to false bc already removed from individual datasets

print(singletonFiltAmpData.shape)

#create the 2D object by reduction analysis of singleton-filtered data
#o=outliers, l=cluster, embedding=data
embedding,o,l=embedding_outliers.fit(singletonFiltAmpData)

#launch app to view interactive bokeh plot of reduced data
mc.plot_umap(embedding,l,ASV_ids,[]) #[] is placeholder bc no taxonomy

#data reduction dataframe
reducedDF=pd.DataFrame()
if len(ASV_ids)>1: #
    reducedDF["ASV_ids"]=ASV_ids.iloc[:,0]
reducedDF["Outliers"]=o
reducedDF["Cluster"]=l

#take a peak
reducedDF.head()

#run SparCC
SparCC_MN.run_all(data_input=dataSparcc)

#save correlations and p-vals- sparcc will compute them separately
DF_SparCC=pd.read_csv(Path(SparCC_MN.save_corr_file).resolve(),index_col=0)
DF_PValues=pd.read_csv(Path(SparCC_MN.outfile_pvals).resolve(),index_col=0)

#find significant correlations
sparcc_corr=DF_SparCC[DF_PValues<0.05].fillna(0)
print(f'The resulting corellation matrix is of size {sparcc_corr.shape}')
sparcc_corr.head()

#Network analysis (network/subgroups/large-scale metrics)
#begin by building the graph of the SparCC matrix
#can normalize correlation values to 0-1 or leave the values as they are. Option 1 recommended.

M = mc.build_network(sparcc_corr) #build network
Mnorm = mc.build_normalize_network(sparcc_corr) #normalize

#create empty network micnet object
NetM=mc.NetWork_MicNet()

#basic network properties
NetM.basic_description(corr=sparcc_corr)

#triad info- different interaction types (+ + + or - + -) using structural balance method
#requires raw data (-1 to 1), not normalized
NetM.structural_balance(M)

#Find communties in network by Louvain method
#(finds clusters based on increasing intragroup interactions and minimizing intergroup interactions)
Communities=NetM.community_analysis(Mnorm)

#extract no. communities and summary of properties/community
print(Communities["Number"])
print(Communities["Community_topology"])
print(Communities["Data"].head())

#centrality analysis
Centrality=NetM.key_otus(Mnorm)

#merge everything into one dataframe
NetDF=pd.DataFrame({"Num_ASVs":Centrality["NUM_OTUS"],
              "Degree_Centrality":Centrality["Degree centrality"],
              "Betweeness_Centrality":Centrality["Betweeness centrality"],
              "Closeness_Centrality":Centrality["Closeness centrality"],
              "PageRank":Centrality["PageRank"],
              "HDBSCAN":reducedDF["Cluster"],
              "Community":Communities["Data"].values.ravel()})

#Bokeh plots
pl = mc.plot_bokeh(graph=M,frame=NetDF,
              nodes = M.number_of_nodes(),
              max = sparcc_corr.max().max(),
              min = sparcc_corr.min().min(),
              kind_network="spring",
              kind="Community",MapTextNo="Num_ASVs")
show(pl)

pl2 = mc.plot_bokeh(graph=M,frame=NetDF,
              nodes = M.number_of_nodes(),
              max = sparcc_corr.max().max(),
              min = sparcc_corr.min().min(),
              kind_network='circular',
              kind='HDBSCAN',MapTextNo="Num_ASVs")
show(pl2)

#large-scale metrics
#assumes underlying topology is:
#1) a random Erdos-Renyi network, built using function nx.erdos_renyi_graph
#2) a small world Watts-Strogatz built using nx.watts_strogatz_graph function, or
#3) a scale-free Barabási-Albert network built using nx.barabasi_albert_graph function.

#correlation matrix as input and returns three dataframes with
#distribution for several large-scale metrics, assuming network with the same density and average degree but with the defined topologies mentioned.

#Run boostrap
df_rand, df_small, df_scale = mc.topology_boostrap(sparcc_corr, n_boot=20)
