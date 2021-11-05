sq.euc.dist= function(x1, x2) sum((x1 - x2) ^ 2)

wt.euc.dist.sq=function(x1,x2,w){
  p=(x1-x2)^2
  p=w*p
  return(sum(p))
}

vec.wt.euc.dist.sq=function(x1,x2,w){
  p=(x1-x2)^2
  p=w*p
  return(p)
}
## install.packages("nortest")
library(nortest)

sq.euc.dist= function(x1, x2) sum((x1 - x2) ^ 2)
k.means= function(X,M,w,tmax){
  X=as.matrix(X)
  
  
  N =(dim(X))[1]
  k=dim(M)[1]
  t=0
  label=numeric(N)
  wdist=numeric(k)
  repeat{
    t=t+1
    
    
    for(i in 1:N){
      for(j in 1:k){
        wdist[j]=wt.euc.dist.sq(X[i,],M[j,],w)
      }
      label[i]=which.min(wdist)
      
    }
    for(i in 1:k){
      I=which(label==i)
      M[i,]=colMeans(X[I,])
    }
    
    
    
    if(t>tmax){
      break
    }
  }
  return(list(label,M))
  
}


proj=function(X,v,w){
  X=as.matrix(X)
  n=dim(X)[1]
  pro=numeric(dim(X)[1])
  norm.v.sq=sum(v*v*w)
  for(i in 1:n){
    pro[i]=sum(X[i,]*v*w)/norm.v.sq
  }
  #    hist(pro)
  pro
}

split=function(c,X,w){
  X=as.matrix(X)
  p=prcomp(X)
  
  #    A=diag(w)
  #    B=A%*%B
  #   B=B%*%A
  #    cat(A)
  s=p$rotation[,1]
  lambda=p$sdev[1]^2
  m=sqrt(2*lambda/pi)*s
  rm(p)
  return(list(c+m,c-m))
}






split.2=function(X,c,alpha,w){
  X=as.matrix(X)
  flag=0
  N=dim(X)[1]
  d=dim(X)[2]
  
  spl=split(c,X,w)
  c1=spl[[1]]
  c2=spl[[2]]
  c1=as.matrix(c1)
  c2=as.matrix(c2)
  M=cbind(c1,c2)
  M=t(M)
  p=k.means(X,M,w,50)
  M=p[[2]]
  c1=M[1,]
  c2=M[2,]
  v=c1-c2
  projected=proj(X,v,w)
  test=ad.test(projected)
  if(test$p.value<alpha){
    flag=1
    return(list(flag,c1,c2))
  }else{
    flag=0
    return(list(flag,c))
  }
}


wgmeans=function(X,beta,alpha,tmax){
  n=dim(X)[1]
  d=dim(X)[2]
  for(i in 1:d){
    X[,i]=X[,i]-mean(X[,i])
    X[,i]=X[,i]/sd(X[,i])
  }
  M=as.matrix(colMeans(X))
  
  M=t(M)
  weight=rep(1/d,d)
  label=rep(1,n)
  #dist=numeric(c)
  t=0
  D=numeric(d)
  flag=0
  #plot(X)
  repeat{
    t=t+1
    flag=0
    #update centres and clusternumbers
    
    
    
    if(is.vector(M)==TRUE){
      M=as.matrix(M)
      M=t(M)
    }
    
    c=dim(M)[1]
    counter=1
    c1=c   
    new.mat=matrix(rep(0,2*c *d),ncol=d)
    #cat(c1)
    for(i in 1 : c){
      I=which(label==i)
      ##    if(length(I)<8){
      ##        break
      
      ##  }
      s=split.2(X[I,],M[i,],alpha,weight^beta)
      
      if(s[[1]]==1){
        new.mat[counter,]=s[[2]]
        new.mat[(counter+1),]=s[[3]]
        counter=counter+2
      }else if(s[[1]]==0){
        new.mat[counter,]=s[[2]]
        counter=counter+1
      }
    }
    
    rm(M)
    M=matrix(rep(0,d * (counter-1)),ncol=d)
    for(i in 1 : (counter-1)){
      M[i,]=new.mat[i,]
    }
    rm(new.mat)        
    c=dim(M)[1]
    #        if(c1==c){
    #           break
    
    #    }
    
    
    ##        p=k.means(X,M,weight^beta,200)
    ##        M=p[[2]]
    ##    points(M,col=3,pch=19)
    
    #cat(c)
    dist=numeric(c)
    for(iter in 1:30){                                        #update membership
      for(i in 1 : n){
        for(j in 1 : c){
          dist[j]=wt.euc.dist.sq(X[i,],M[j,],weight^beta)
        }
        label[i]=which.min(dist)
      }
      
      #update centres
      
      for(i in 1:c){
        I=which(label==i)
        M[i,]=colMeans(X[I,])
      }
      
      #update weights
      
      
      for(j in 1:d){
        D[j]=0
      }
      for(i in 1:c){
        I=which(label==i)
        for(k in I){
          D=D+vec.wt.euc.dist.sq(X[k,],M[i,],rep(1,d))
        }
      }
      
      for(i in 1:d){
        if(D[i]!=0){
          D[i]=1/D[i]
          D[i]=D[i]^(1/(beta-1))
        }
      }
      s=sum(D)
      weight=D/s
      
      
      
    }
    #  cat(weight)
    # cat('\n')
    ##          cat(weight)
    #check to discard features
    ##        max=max(weight)
    ##      max=max*.1
    ##    truth=(weight<max)
    ##  if(sum(truth)>0){
    ##I=which(weight<max)
    
    ##  weight(I)=0
    ## }
    if(t>tmax){
      break
      cat('here1')
    }
    for(i in 1:c){
      I=which(label==i)
      if(length(I)<8){
        flag=1
      }
    }
    if(flag==1){
      break
      cat('here2')
    }
    if(c1==c){

      break
      cat('here3')
    }
    
    
    
  }
  return(list(label,M,weight,t))
}

