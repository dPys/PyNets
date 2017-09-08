library("viridis")
library("zoom")
library("pvclust")
library("purrr")
#analyzing HBN Data
#source("/Users/aki.nikolaidis/Dropbox/1_Projects/1_Research/2_FosterCare/Code/functions.r")
#TwoWayclust(WIAT_NIH7,group_thresh = 1.5,"/Users/aki.nikolaidis/Dropbox/1_Projects/2_Grants/NARSAD/HBN_Data")

data<-read.csv('/Users/aki.nikolaidis/git_repo/PyNets/Testing/dim_reduce.csv',row.names = 1, header = TRUE, sep = ",");
pynets_path<- '/Users/aki.nikolaidis/git_repo/PyNets'

UnprocessedData <- data
UnprocessedData[is.na(UnprocessedData) == TRUE] <- 0
sum(is.na(UnprocessedData))

Clusterset <- data.frame(UnprocessedData)

Clusterset <- as.matrix(Clusterset);
Clusterset <- scale(Clusterset)  
#OutlierID <- abs(Clusterset)<4
#subs <- apply(OutlierID, 1, min)
#Clusterset <- Clusterset[subs==1,]

#3.2 Euclidean + Ward Clustering of Subjects
r_dist <- dist(Clusterset, method = "euclidean")
hr <- hclust(r_dist, method = "ward.D2");

#3.3- Spearman + Complete Clustering of Variables
hc <- hclust(as.dist(1 - cor(Clusterset, method = "spearman")), method = "complete");

#4-Subject Group Assignment
Sub_Group <- cutree(hr, h = max(hr$height)/1.5)
mycolhr <- rainbow(length(unique(Sub_Group)), start = 0.1, end = 0.9); 
mycolhr <- mycolhr[as.vector(Sub_Group)]

#5-Variable Group Assignment
Var_Group <- cutree(hc, h = max(hc$height)/1.1)
mycolhc <- rainbow(length(unique(Var_Group)), start = 0.1, end = 0.9); 
mycolhc <- mycolhc[as.vector(Var_Group)]

#6- Visualization
#FIX SIZE OF SAVED FILE
file_location<- paste(pynets_path,'Testing/heatmap.jpg', sep = "")
jpeg(file = file_location)
heatmap(Clusterset, Rowv = as.dendrogram(hr), Colv = as.dendrogram(hc), col = inferno(256), scale = "none", RowSideColors = mycolhr, ColSideColors = mycolhc)
dev.off()
#-------------------------------------------------------------------------------------
