# make-synthetic-wa.R
# generate synthetic mobility data for Washington State CBGs
# Synthetic data is based on 4 basis functions.
# A timeseries is generated for each CBG.
# The timeseries of a CBGs generated using either a single basis function or a combination of 2.
# CBGs within each county use the same basis function(s).

library(sf)

censusdatadirectory <- "../data/"  # census data

statename <- "washington"
statefips <- 53
stateabbr <- "wa"

# load cbg shapefile
shapefile.cbg <- read_sf(paste(censusdatadirectory,"cb_2019_",statefips,"_bg_500k.shp",sep=""))

cbglist <- shapefile.cbg$GEOID
dateslist <- as.Date("2020/02/23") + 0:120

numbasisfunctions <- 4
bases <- matrix(NA, ncol=length(dateslist), nrow=numbasisfunctions)
bases[1,] <- (1+sin((1:ncol(bases)/7)))/2
bases[2,] <- (1+sin(5+(1:ncol(bases)/12)))/2
bases[3,] <- exp(-(1:ncol(bases)/30))
bases[4,] <- floor((1:ncol(bases))/50) %% 2

plot(x=NA, y=NA, xlim=c(0,ncol(bases)), ylim=c(0,1), main="basis functions", xlab="", ylab="stay at home")
for (i in 1:nrow(bases)) {
    lines(x=1:ncol(bases), y=bases[i,], col=i, lwd=3)
}

data <- matrix(NA, ncol=length(dateslist), nrow=length(cbglist))
rownames(data) <- cbglist
colnames(data) <- format(dateslist, "%Y/%m/%d")

# four counties have counts based on one basis function
data[substr(rownames(data),1,5)=="53033",] <- rep(bases[1,],each=sum(substr(rownames(data),1,5)=="53033")) * rep(0.1+0.9*runif(sum(substr(rownames(data),1,5)=="53033")),times=ncol(data)) # King County is basis 1
data[substr(rownames(data),1,5)=="53011",] <- rep(bases[2,],each=sum(substr(rownames(data),1,5)=="53011")) * rep(0.1+0.9*runif(sum(substr(rownames(data),1,5)=="53011")),times=ncol(data)) # Clark is basis 2
data[substr(rownames(data),1,5)=="53063",] <- rep(bases[3,],each=sum(substr(rownames(data),1,5)=="53063")) * rep(0.1+0.9*runif(sum(substr(rownames(data),1,5)=="53063")),times=ncol(data)) # Spokane is basis 3
data[substr(rownames(data),1,5)=="53047",] <- rep(bases[3,],each=sum(substr(rownames(data),1,5)=="53047")) * rep(0.1+0.9*runif(sum(substr(rownames(data),1,5)=="53047")),times=ncol(data)) # Okanagan is basis 4

# the remaining counties have counts based on a combination of two basis functions
for (i in which(is.na(data[,1]))) {
    countyfips <- substr(rownames(data)[i],1,5)
    weights <- c(as.numeric(substr(rownames(data)[i],6,8))/1000,as.numeric(substr(rownames(data)[i],9,11))/1000) # weights of the 2 basis functions are based on digits of the CBG FIPS
    if (as.numeric(countyfips)%%3==0) {
        # combination of bases 1 and 2
        data[i,] <- (bases[1,]+weights[1])*(bases[2,]+weights[2])
    } else if (as.numeric(countyfips)%%3==1) {
        # combination of bases 2 and 3
        data[i,] <- (bases[2,]+weights[1])*(bases[3,]+weights[2])
    } else if (as.numeric(countyfips)%%3==2) {
        # combination of bases 3 and 4
        data[i,] <- (bases[3,]+weights[1])*(bases[4,]+weights[2])
    }
}

# add noise
noise <- matrix(rnorm(ncol(data)*nrow(data), mean=0.4, sd=0.1),ncol=ncol(data),nrow=nrow(data))
noise[noise<0] <- 0
data <- data+noise

# normalize to 1
data <- data/(apply(data, 1, max))

# write out data
write.csv(round(data,4), file="synthetic-stayathome-washington.csv", row.names=TRUE, quote=FALSE)
