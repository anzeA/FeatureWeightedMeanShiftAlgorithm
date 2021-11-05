#!/usr/bin/env Rscript
# Example of running this code
# Rscript.exe --vanilla .\wgmeans_run.R data/test_data.csv 2 0.0001 data/test_out.csv
# β is the exponent of the weights
#make sure data are scaled
source('wgmeans.R')
args <- commandArgs(trailingOnly=TRUE)#

# test if there is at least one argument: if not, return an error

if (length(args)!=4) {
  stop("At least three argument must be supplied input file, lambda, output file ", call.=FALSE)
  q()
}
input_file <- args[1]#'data\\tmp_data.csv'
beta <- as.numeric(args[2])
alpha <- as.numeric(args[3])
output_file <- args[4]



data <- read.csv(file = input_file) #'data/test_data.csv'
X <- data.matrix(data)
clusters <- wgmeans(X, beta=beta, alpha = alpha, tmax=100)
clusters <- as.numeric(clusters[[1]])-1

#clusters
data_new <- cbind(data, clusters) # need to shift to get from index 0
#data_new
write.csv(data_new,output_file, row.names = FALSE)
q()