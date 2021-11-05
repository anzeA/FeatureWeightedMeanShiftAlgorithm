source('WBMS.R') # Read the Functions
args <- commandArgs(trailingOnly=TRUE)#

# test if there is at least one argument: if not, return an error
if (length(args)!=5) {
  stop("At least three argument must be supplied input file, lambda, output file ", call.=FALSE)
  q()
}
setwd('D://GitRepositories//FeatureWeightedMeanShiftAlgorithm')

input_file <- args[1]#'data\\tmp_data.csv'
h <- as.numeric(args[2])
lambda <- as.numeric(args[3])
output_file <- args[4]
output_file_weights <- args[5]

#setwd("D://GitRepositories//FeatureWeightedMeanShiftAlgorithm")

data <- read.csv(file = input_file) #'data/test_data.csv'
#data <- read.csv(file = 'data/test_data.csv')
#data <- read.csv(file = 'data/tmp_wbms.csv')
#data <- read.csv(file = 'data//toy_data.csv') #'data/test_data.csv'
#data
X <- data.matrix(data)
l=WBMS(X,h,lambda ,tmax=10)
#l=WBMS(X,0.1,5 ,tmax=5)
clusters = U2clus(l[[1]])
clusters <- as.numeric(clusters[[1]])-1

#clusters
data_new <- cbind(data, clusters)
#data_new
write.csv(data_new,output_file, row.names = FALSE)
write.csv(l[[2]],output_file_weights, row.names = FALSE)
q()

