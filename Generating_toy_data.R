rm(list=ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Function
GenerateToyData=function(x,n=1000,p=NULL,seed=1,observation_consistency=FALSE){
  # x is the original (real) dataset
  # By default (p=NULL), all variables will be kept in the resulting toy example dataset
  # You can choose how many observations to keep in the toy example dataset (no need to go over 100 in general)
  # You can change the seed if you want to generate multiple toy example datasets (not needed in general)
  
  if (is.null(p)){
    p=ncol(x)
  }
  
  if (n>nrow(x)){
    stop(paste0("Please set n to be smaller than the number of rows in the original dataset (i.e. n<=", nrow(x), ")"))
  }
  
  # Random sub-sample of rows and columns
  set.seed(seed)
  s=sample(nrow(x), size=n)
  if (p==ncol(x)){
    xtoy=x[s,1:p]
  } else {
    xtoy=x[s,sample(p)]
  }
  
  if (!observation_consistency){
    # Permutation of the observations by variable (keeping the structure of missing data)
    for (k in 1:p){
      xtmp=xtoy[,k]
      xtmp[!is.na(xtmp)]=sample(xtmp[!is.na(xtmp)])
      xtoy[,k]=xtmp
    }
  }
  
  rownames(xtoy)=1:nrow(xtoy)
  return(xtoy)
}


# Loading the original (real) data
data=read.csv("US_births_2018.csv")

# Make sure that ID is not one of the variables
#rownames(data)=data$eid
#data=data[,-1]

# Generate a toy example dataset with all variables and a subset of 100 observations
toy_data=GenerateToyData(x=data, seed=1)
write.csv(toy_data, "Births_toy.csv")

# Change the seed to generate the covariate toy example dataset
#toy_covars=GenerateToyData(x=covars, seed=2, observation_consistency=TRUE)
#saveRDS(toy_covars, "Covars_toy.rds")

# The observation_consistency argument prevents re-shuffling within the variables so that the dataset is consistent
#table(toy_covars$vit_status, toy_covars$dc_cancer_st)
# Example: people who died of cancer are also coded as dead in vit_status

# You can then run models on the toy example data, e.g. 
print(all(rownames(toy_biomarkers)==rownames(toy_covars)))
bmk=toy_biomarkers[,1]
y=toy_covars$vit_status
model=glm(y~bmk, family="binomial")
summary(model)
# You should not find much significant associations due to the permutations...



