#!/usr/bin/env Rscript
# Example of running this code
# Rscript.exe --vanilla .\EWDP.R data/test_data.csv 1 1 data/test_out.csv


## install.packages("devtools")
## library(devtools)
## devtools::install_github("DebolinaPaul/EWDP")
library("EWDPmeans")

#make sure data are scaled



args <- commandArgs(trailingOnly=TRUE)#

# test if there is at least one argument: if not, return an error

if (length(args)!=4) {
  stop("At least three argument must be supplied input file, lambda, output file ", call.=FALSE)
  q()
}
#library(maotai)
input_file <- args[1]
lambda_w <- as.numeric(args[2])
lambda_k <- as.numeric(args[3])
output_file <- args[4]
data <- read.csv(file = input_file) #'data/test_data.csv'
X <- data.matrix(data)

# lambda_w=1, lambda_k=1 defoult values
clusters <- EWDPmeans(X, lambda_w=lambda_w, lambda_k=lambda_k, tmax=50)
clusters <- as.numeric(clusters[[3]])-1
data_new <- cbind(data, clusters) # need to shift to get from index 0
#data_new
write.csv(data_new,output_file, row.names = FALSE)
q()

#Run the code


#Number of clusters
l[[1]]
#Feature weights
l[[2]]
#Cluster labels
l[[3]]