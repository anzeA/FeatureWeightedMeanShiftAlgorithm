## install.packages("maotai")
#!/usr/bin/env Rscript

# Example of running this code
# Rscript.exe --vanilla .\DP-means.R data/test_data.csv 5 data/test_out.csv

args <- commandArgs(trailingOnly=TRUE)#

# test if there is at least one argument: if not, return an error

if (length(args)!=3) {
  stop("At least three argument must be supplied input file, lambda, output file ", call.=FALSE)
  q()
}
library(maotai)
input_file <- args[1]
lambda <- as.numeric(args[2])
output_file <- args[3]
data <- read.csv(file = input_file) #'data/test_data.csv'
X <- data.matrix(data)


clusters <- dpmeans(X, lambda= lambda)$cluster
clusters <- as.numeric(clusters)-1
data_new <- cbind(data, clusters) # need to shift to get from index 0
write.csv(data_new,output_file, row.names = FALSE)
q()
##---------------------------------------------------------------------------------


## define data matrix of two clusters
## x1  = matrix(rnorm(50*3,mean= 2), ncol=3)
##x2  = matrix(rnorm(50*3,mean=-2), ncol=3)
# X   = rbind(x1,x2)
##lab = c(rep(1,50),rep(2,50))
## run dpmeans with several lambda values

solA <- dpmeans(X, lambda= 5)$cluster
solB <- dpmeans(X, lambda=10)$cluster
solC <- dpmeans(X, lambda=20)$cluster
## visualize the results
opar <- par(no.readonly=TRUE)
par(mfrow=c(1,4), pty="s")
plot(X,col=lab,  pch=19, cex=.8, main="True", xlab="x", ylab="y")
plot(X,col=solA, pch=19, cex=.8, main="dpmeans lbd=5", xlab="x", ylab="y")
plot(X,col=solB, pch=19, cex=.8, main="dpmeans lbd=10", xlab="x", ylab="y")
plot(X,col=solC, pch=19, cex=.8, main="dpmeans lbd=20", xlab="x", ylab="y")
par(opar)


## let's find variations by permuting orders of update
## used setting : lambda=20, we will 8 runs
sol8 <- list()
for (i in 1:8){
  sol8[[i]] = dpmeans(X, lambda=20, permute.order=TRUE)$cluster
}

## let's visualize
vpar <- par(no.readonly=TRUE)
par(mfrow=c(2,4), pty="s")
for (i in 1:8){
  pm = paste("permute no.",i,sep="")
  plot(X,col=sol8[[i]], pch=19, cex=.8, main=pm, xlab="x", ylab="y")
}
par(vpar)