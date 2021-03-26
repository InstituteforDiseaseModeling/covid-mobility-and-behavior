# make-synthetic-wa.R
# generate synthetic mobility data for Washington State CBGs
# A timeseries is generated for each CBG.
# Synthetic data is based on 4 basis functions.
# The timeseries of a CBGs generated using either a single basis function or a combination of 2.
# CBGs within a county use the same basis function(s).
# Will output plots in the working directory.

library(sf)

censusdatadirectory <- "../data/"  # census data
syntheticdatadirectory <- ""  # output directory for synthetic data
statename <- "washington"
statefips <- 53

set.seed(1) # set random number generator seed for replicability

# load shapefiles
shapefile.cbg <- read_sf(paste(censusdatadirectory,"cb_2019_",statefips,"_bg_500k.shp",sep=""))
shapefile.counties <- read_sf(paste(censusdatadirectory,"cb_2018_us_county_20m.shp",sep=""))

# initialize synthetic data matrix
cbglist <- shapefile.cbg$GEOID
dateslist <- as.Date("2020/02/23") + 0:116 # simulate 117 days of data
data <- matrix(NA, ncol=length(dateslist), nrow=length(cbglist))
rownames(data) <- cbglist
colnames(data) <- format(dateslist, "%Y/%m/%d")

# make time series for the 4 "basis" functions
numbasisfunctions <- 4
bases <- matrix(NA, ncol=length(dateslist), nrow=numbasisfunctions)
bases[1,] <- (1+sin((1:ncol(bases)/7)))/2
bases[2,] <- exp(-(1:ncol(bases)/30))
bases[3,] <- (1+sin(5+(1:ncol(bases)/12)))/2
bases[4,] <- exp(-((ncol(bases):1)/30))

# four counties have time series based on one basis function
anchors <- c("53063","53033","53047","53011")
data[substr(rownames(data),1,5)=="53063",] <- rep(bases[1,],each=sum(substr(rownames(data),1,5)=="53063")) * rep(0.2+0.8*runif(sum(substr(rownames(data),1,5)=="53063")),times=ncol(data)) # Spokane is basis 1
data[substr(rownames(data),1,5)=="53033",] <- rep(bases[2,],each=sum(substr(rownames(data),1,5)=="53033")) * rep(0.2+0.8*runif(sum(substr(rownames(data),1,5)=="53033")),times=ncol(data)) # King County is basis 2
data[substr(rownames(data),1,5)=="53047",] <- rep(bases[3,],each=sum(substr(rownames(data),1,5)=="53047")) * rep(0.2+0.8*runif(sum(substr(rownames(data),1,5)=="53047")),times=ncol(data)) # Okanagan is basis 3
data[substr(rownames(data),1,5)=="53011",] <- rep(bases[4,],each=sum(substr(rownames(data),1,5)=="53011")) * rep(0.2+0.8*runif(sum(substr(rownames(data),1,5)=="53011")),times=ncol(data)) # Clark is basis 4

# the CBGs in remaining counties have time series based on a combination of two basis functions
for (i in which(is.na(data[,1]))) {
    countyfips <- substr(rownames(data)[i],3,5)
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

# add noise to all time series
noise <- matrix(rnorm(ncol(data)*nrow(data), mean=0.4, sd=0.1),ncol=ncol(data),nrow=nrow(data))
noise[noise<0] <- 0
data <- data+noise

# normalize to 0.7 (similar range as SafeGraph "stay at home" data for plotting)
data <- 0.7*data/(apply(data, 1, max))

# write out data
write.csv(round(data,4), file=paste(syntheticdatadirectory,"synthetic-stayathome-washington.csv",sep=""), row.names=TRUE, quote=FALSE)

# basis functions
plot(x=NA, y=NA, xlim=c(0,ncol(bases)), ylim=c(0,1.0), main="basis functions", xlab="", ylab="")
for (i in 1:nrow(bases)) {
    lines(x=1:ncol(bases), y=bases[i,], col=i, lwd=3)
}

# plot a few samples from the 4 counties based on one basis function
png("synthtimeseries-4bases.png", width=800, height=1200)
par(mfrow=c(4,1),
    mar=c(2,4,2,0.5), #bottom, left, top, and right.
    mgp=c(2.5,0.6,0))
for (i in 1:4) {
    plot(x=1:ncol(data), y=data[which(substr(rownames(data),1,5)==anchors[i]),][1,], ylim=c(0,0.7), main=paste("5 CBGs from basis function",i), ylab="stay at home", xlab="", cex.main=2, cex.lab=1.4, cex.axis=1.4)
    samples <- floor(runif(5)*sum(substr(rownames(data),1,5)==anchors[i]))+1
    for (j in 1:5) {
        lines(x=1:ncol(data), y=data[which(substr(rownames(data),1,5)==anchors[i]),][samples[j],], col=j+1)
    }
}
dev.off()

# plot a few samples from the 4 counties based on two basis functions
png("synthtimeseries-3combos.png", width=800, height=900)
par(mfrow=c(3,1),
    mar=c(2,4,2,0.5), #bottom, left, top, and right.
    mgp=c(2.5,0.6,0))
v <- shapefile.counties$STATEFP=="53"
for (i in 1:3) {
    countylist <- shapefile.counties$COUNTYFP[v] # all washington counties
    countylist <- countylist[!(countylist %in% substr(anchors,3,5))]
    countylist <- countylist[(as.numeric(countylist)%%3)==(i-1)]
    v2 <- (substr(rownames(data),3,5) %in% countylist)
    plot(x=1:ncol(data), y=data[which(v2),][1,], ylim=c(0,0.7), main=paste("5 CBGs from basis function",i,"+",i+1), ylab="stay at home", xlab="", cex.main=2, cex.lab=1.4, cex.axis=1.4)
    samples <- floor(runif(5)*sum(v2))+1
    for (j in 1:5) {
        lines(x=1:ncol(data), y=data[v2,][samples[j],], col=j+1)
    }
}
dev.off()

# maps
png("synthmap-counties.png", width=800, height=600)
pal <- c("lightblue","seagreen","salmon")
v <- shapefile.counties$STATEFP=="53"
plot(st_geometry(shapefile.counties[v,]), col=ifelse(shapefile.counties$COUNTYFP[v] %in% substr(anchors,3,5),NA,pal[1+as.numeric(shapefile.counties$COUNTYFP[v])%%3]), main="Basis functions by county", cex.main=2.5)
centroids <- st_centroid(shapefile.counties[v,])
labpts <- st_coordinates(centroids)
text(labpts[,1], labpts[,2], 
     lab=ifelse(shapefile.counties$COUNTYFP[v] %in% substr(anchors,3,5),match(shapefile.counties$COUNTYFP[v],substr(anchors,3,5)),
                c("1*2","2*3","3*4")[1+as.numeric(shapefile.counties$COUNTYFP[v])%%3]), col="darkblue", cex=2)
for (i in 1:length(anchors)) { # draw the shape of the basis functions
    lines(x=labpts[which(shapefile.counties$COUNTYFP[v]==substr(anchors[i],3,5)),1]+0.005*(1:ncol(bases))-0.25, y=labpts[which(shapefile.counties$COUNTYFP[v]==substr(anchors[i],3,5)),2]+0.1*(bases[i,])-0.16, col="darkred", lwd=1)
}
dev.off()
