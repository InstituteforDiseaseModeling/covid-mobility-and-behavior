# make-census-plots.R
# makes plots for the clustering paper

censusdatadirectory <- "../data/"  # census data
safegraphdatadirectory <- "../data/" # safegraph data
clusterdatadirectory <- "../data/" # all other data
figuredirectory <- "" # output directory for figures

#####################################################
# Make plots for manuscript
library(sf)
library(jsonlite)
library(dplyr)
library(data.table)
library(clinfun) # for Jonckheere-Terpstra test

pal.cluster <- c("#ff7f00","#fdbf6f","#a6cee3","#1f78b4","#cc78bc")
names(pal.cluster) <- c("A","B","C","D","E")
pal.cluster.dark <- pal.cluster
names(pal.cluster.dark) <- c("A","B","C","D","E")

for (region in c("washington","texas","california","georgia")) {
    print(region)
    statename <- region
    statefips <- ifelse(region %in% c("king","yakima","washington"),"53",ifelse(region %in% c("la","california"),"06",ifelse(region %in% c("atlanta","georgia"),"13",ifelse(region %in% c("denver"),"08",ifelse(region=="florida","12",ifelse(region=="texas","48",ifelse(region=="idaho","16",NULL)))))))

# Get cluster data
# load transformed and cluster data
    data.clusters <- list()
    if (statename=="washington") { # different file names used for WA state
        data.clusters[["gmm5"]] <- read.csv(paste(clusterdatadirectory,"labels_GMM_on_SE_5clust_seed0.csv",sep=""), sep=",", header=TRUE)
        data.clusters[["gmm5"]]$cluster <- c("B","D","E","A","C")[data.clusters[["gmm5"]]$x]
    } else {
        data.clusters[["gmm5"]] <- read.csv(paste(clusterdatadirectory,"labels_",stateabbr,".csv",sep=""), sep=",", header=TRUE)
        colnames(data.clusters[["gmm5"]]) <- c("ind","x")
        if (statename=="texas") {
            data.clusters[["gmm5"]]$cluster <- c("A","B","C","D","E")[data.clusters[["gmm5"]]$x]
        } else if (statename=="georgia") {
            data.clusters[["gmm5"]]$cluster <- c("D","C","B","A","E")[data.clusters[["gmm5"]]$x]
        } else if (statename=="california") {
            data.clusters[["gmm5"]]$cluster <- c("D","C","B","A","E")[data.clusters[["gmm5"]]$x]
        } else {
            data.clusters[["gmm5"]]$cluster <- c("D","C","A","B","E")[data.clusters[["gmm5"]]$x]
        }
    }

# load census data
    censusdata <- readRDS(file=filename <- paste(censusdatadirectory,"census-forclustering-",statename,".rds",sep=""))
    pop.acs2018 <- censusdata[[paste(statename,"b01001")]][censusdata[[paste(statename,"b01001")]]$variable=="B01001_001",]$estimate
    names(pop.acs2018) <- censusdata[[paste(statename,"b01001")]][censusdata[[paste(statename,"b01001")]]$variable=="B01001_001",]$GEOID
    data.income <- censusdata[[paste(statename,"b19013_001")]][censusdata[[paste(statename,"b19013_001")]]$variable=="B19013_001",]$estimate
    names(data.income) <- censusdata[[paste(statename,"b19013_001")]][censusdata[[paste(statename,"b19013_001")]]$variable=="B19013_001",]$GEOID

# load cbg shapefile
    shapefile.cbg <- read_sf(paste(censusdatadirectory,"cb_2019_",statefips,"_bg_500k.shp",sep="")) # from https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html
#if (substr(shapefile.cbg$GEOID[1],1,1)=="0") {
#    shapefile.cbg$GEOID <- substr(shapefile.cbg$GEOID,2,12) # chop of leading 0s
#}

# load processed SafeGraph data
    data.stayathome <- read.csv(paste(safegraphdatadirectory,"stayathome-",statename,".csv",sep=""), sep=",", header=TRUE)
    data.devicecount <- read.csv(paste(safegraphdatadirectory,"devicecount-",statename,".csv",sep=""), sep=",", header=TRUE)
    colnames(data.stayathome)[1] <- "cbg"
    colnames(data.devicecount)[1] <- "cbg"
    colnames(data.stayathome) <- gsub("^X","",colnames(data.stayathome))
    colnames(data.stayathome) <- gsub("[.]","/",colnames(data.stayathome))
    colnames(data.devicecount) <- gsub("^X","",colnames(data.devicecount))
    colnames(data.devicecount) <- gsub("[.]","/",colnames(data.devicecount))

# load raw SafeGraph data
    data.socialdistancing <- readRDS(file=paste(safegraphdatadirectory,"socialdistancing-",statename,".rds",sep=""))
    dates.socialdistancing <- names(data.socialdistancing)[grepl(paste(statename," ",sep=""),names(data.socialdistancing))]
    dates.socialdistancing <- gsub(paste(statename," ",sep=""),"",dates.socialdistancing)
    dates.socialdistancing <- dates.socialdistancing[as.Date(dates.socialdistancing)>="2020/02/23" & as.Date(dates.socialdistancing)<"2020/09/01"] # only keep dates from 2/23/2020 and later

###########################
# census variable associations

    png(paste(figuredirectory,"boxplot-gmm5-",statename,".png",sep=""), width=1400, height=360)
    capstatename <- paste(toupper(substr(statename,1,1)),substr(statename,2,100),sep="")
    par(mfrow=c(1,3),
        mar=c(3.8,7.5,3,0.8), #bottom, left, top, and right.
        mgp=c(4.65,1.2,0))
    for (i in 1:3) {
        ylim <- c(0,15)
        plotlog <- FALSE
        ylab <- NA
        if (i==1) {
            v <- c(FALSE, as.Date(colnames(data.stayathome)[-1]) >= "2020/02/23" & as.Date(colnames(data.stayathome)[-1]) <= "2020/06/18") # 117 days of data
            meanstay <- rowMeans(data.stayathome[,v], na.rm=TRUE)
            temp <- 100*meanstay[match(as.character(data.clusters[["gmm5"]]$ind), data.stayathome$cbg)]
            title <- paste("Mean stay-at-home,", capstatename)
            ylab <- "% staying at home"
            ylim <- c(0,60)
        } else if (i==2) {
            temp <- pop.acs2018[match(as.numeric(data.clusters[["gmm5"]]$ind),as.numeric(names(pop.acs2018)))] / shapefile.cbg$ALAND[match(as.numeric(data.clusters[["gmm5"]]$ind), as.numeric(shapefile.cbg$GEOID))]
            temp[temp==0] <- NA # don't plot on log scale
            title <- paste("2018 population density,", capstatename)
            ylab <- "people/sq meter"
            plotlog <- TRUE
            ylim <- c(2e-8, 1e-1)
        } else if (i==3) {
            temp <- data.income[match(as.character(data.clusters[["gmm5"]]$ind), as.character(names(data.income)))]
            title <- paste("2018 median income,", capstatename)
            ylab <- "$ per year"
            ylim <- c(0,250000)
        }
        result <- jonckheere.test(x=temp, g=match(data.clusters[["gmm5"]]$cluster,c("A","B","C","D")), alternative = "decreasing", nperm=1000)
        print(result)
                                        #    result <- jonckheere.test(x=temp, g=match(data.clusters[["gmm5"]]$cluster,c("A","B","C","D")), alternative = "increasing", nperm=1000)
                                        #    print(result)
        result <- jonckheere.test(x=temp, g=match(data.clusters[["gmm5"]]$cluster,c("B","C","D")), alternative = "decreasing", nperm=1000) # for California
        print(result)

        if (plotlog) {
            boxplot(temp[data.clusters[["gmm5"]]$cluster=="A"],
                    temp[data.clusters[["gmm5"]]$cluster=="B"],
                    temp[data.clusters[["gmm5"]]$cluster=="C"],
                    temp[data.clusters[["gmm5"]]$cluster=="D"],
                    temp[data.clusters[["gmm5"]]$cluster=="E"],
                    main=title, ylab=ylab,
                    names=rep(NA,5),
                    col=pal.cluster, cex.main=2.9, cex.axis=2.9, cex.lab=3.6,
                    log="y", ylim=ylim)
            axis(1, at=1:5, lab=c("A","B","C","D","E"), cex.axis=3.5, padj=0.45)
        } else {
            boxplot(temp[data.clusters[["gmm5"]]$cluster=="A"],
                    temp[data.clusters[["gmm5"]]$cluster=="B"],
                    temp[data.clusters[["gmm5"]]$cluster=="C"],
                    temp[data.clusters[["gmm5"]]$cluster=="D"],
                    temp[data.clusters[["gmm5"]]$cluster=="E"],
                    main=title, ylab=ylab,
                    names=rep(NA,5),
                                        #                names=c("A","B","C","D","E"),
                    col=pal.cluster, cex.main=2.9, cex.axis=2.9, cex.lab=3.6,
                    ylim=ylim)
            axis(1, at=1:5, lab=c("A","B","C","D","E"), cex.axis=3.5, padj=0.45)
        }
    }
    dev.off()

    # might not use this figure
    png(paste(figuredirectory,"scatter-gmm5-",statename,".png",sep=""), width=1200, height=600)
    capstatename <- paste(toupper(substr(statename,1,1)),substr(statename,2,100),sep="")
    par(mfrow=c(1,2),
        mar=c(5.8,5.8,3,0.8), #bottom, left, top, and right.
        mgp=c(3.9,1.2,0))
    for (i in 1:2) {
        ylim <- c(0,15)
        plotlog <- FALSE
        ylab <- NA
        v <- c(FALSE, as.Date(colnames(data.stayathome)[-1]) >= "2020/02/23" & as.Date(colnames(data.stayathome)[-1]) <= "2020/06/18") # 117 days of data
        meanstay <- rowMeans(data.stayathome[,v], na.rm=TRUE)
        xdata <- 100*meanstay[match(as.character(data.clusters[["gmm5"]]$ind), data.stayathome$cbg)]
        title <- paste("Mean stay-at-home,", capstatename)
        xlab <- "% staying at home"
        xlim <- c(10,65)
        if (i==1) {
            ydata <- data.income[match(as.character(data.clusters[["gmm5"]]$ind), names(data.income))]
            title <- paste("2018 median income,", capstatename)
            ylab <- "$ per year"
            ylim <- c(0,250000)
        } else if (i==2) {
            ydata <- pop.acs2018[match(as.numeric(data.clusters[["gmm5"]]$ind),as.numeric(names(pop.acs2018)))] / shapefile.cbg$ALAND[match(as.numeric(data.clusters[["gmm5"]]$ind), as.numeric(shapefile.cbg$GEOID))]
            ydata[ydata==0] <- NA # don't plot on log scale
            title <- paste("2018 population density,", capstatename)
            ylab <- "people/sq meter"
            plotlog <- TRUE
            ylim <- c(2e-8, 1e-1)
        }
        plot(x=xdata, y=ydata, xlab=xlab, main=capstatename, ylab=ylab, xlim=xlim, ylim=ylim, col=pal.cluster[data.clusters[["gmm5"]]$cluster], pch=4, cex.main=2.5, cex.axis=1.8, cex.lab=2.4, cex=0.5, lwd=0.5, log=ifelse(plotlog,"y",""))
    }
    dev.off()

######################################
# proportions by geographic mobility

    total1 <- (censusdata[[paste(statename,"b07201")]][censusdata[[paste(statename,"b07201")]]$variable=="B07201_001",]$estimate) # total
    total2 <- (censusdata[[paste(statename,"b07202")]][censusdata[[paste(statename,"b07202")]]$variable=="B07202_001",]$estimate) # total
    total3 <- (censusdata[[paste(statename,"b07203")]][censusdata[[paste(statename,"b07203")]]$variable=="B07203_001",]$estimate) # total
    total <- total1 + total2 + total3
    same1 <- (censusdata[[paste(statename,"b07201")]][censusdata[[paste(statename,"b07201")]]$variable=="B07201_002",]$estimate) # same house 1 year ago
    same2 <- (censusdata[[paste(statename,"b07202")]][censusdata[[paste(statename,"b07202")]]$variable=="B07202_002",]$estimate) # same house 1 year ago
    same3 <- (censusdata[[paste(statename,"b07203")]][censusdata[[paste(statename,"b07203")]]$variable=="B07203_002",]$estimate) # same house 1 year ago
    same <- same1 + same2 + same3
    fracsame <- same/total # fraction in the same house as 1 year ago
    names(fracsame) <- (censusdata[[paste(statename,"b07201")]][censusdata[[paste(statename,"b07201")]]$variable=="B07201_001",]$GEOID)

    grad <- (censusdata[[paste(statename,"b14002_019")]]$estimate + censusdata[[paste(statename,"b14002_043")]]$estimate + censusdata[[paste(statename,"b14002_022")]]$estimate + censusdata[[paste(statename,"b14002_046")]]$estimate) # enrolled in undergrad or grad (M+F)
    names(grad) <- censusdata[[paste(statename,"b14002_019")]]$GEOID
    fracgrad <- grad/pop.acs2018[match(names(grad), names(pop.acs2018))]

    fracrenter <- (censusdata[[paste(statename,"b25008_003")]]$estimate/censusdata[[paste(statename,"b25008_001")]]$estimate) # frac renter
    names(fracrenter) <- censusdata[[paste(statename,"b25008_001")]]$GEOID

    total <- (censusdata[[paste(statename,"b01001")]][censusdata[[paste(statename,"b01001")]]$variable=="B01001_001",]$estimate) # total
    young <- (censusdata[[paste(statename,"b01001")]][censusdata[[paste(statename,"b01001")]]$variable=="B01001_007",]$estimate) + # 18-19
        (censusdata[[paste(statename,"b01001")]][censusdata[[paste(statename,"b01001")]]$variable=="B01001_008",]$estimate) + # 20
        (censusdata[[paste(statename,"b01001")]][censusdata[[paste(statename,"b01001")]]$variable=="B01001_009",]$estimate) + # 21
        (censusdata[[paste(statename,"b01001")]][censusdata[[paste(statename,"b01001")]]$variable=="B01001_010",]$estimate) + # 22-24
        (censusdata[[paste(statename,"b01001")]][censusdata[[paste(statename,"b01001")]]$variable=="B01001_011",]$estimate) + # 25-29
        (censusdata[[paste(statename,"b01001")]][censusdata[[paste(statename,"b01001")]]$variable=="B01001_031",]$estimate) + # 18-19
        (censusdata[[paste(statename,"b01001")]][censusdata[[paste(statename,"b01001")]]$variable=="B01001_032",]$estimate) + # 20
        (censusdata[[paste(statename,"b01001")]][censusdata[[paste(statename,"b01001")]]$variable=="B01001_033",]$estimate) + # 21
        (censusdata[[paste(statename,"b01001")]][censusdata[[paste(statename,"b01001")]]$variable=="B01001_034",]$estimate) + # 22-24
        (censusdata[[paste(statename,"b01001")]][censusdata[[paste(statename,"b01001")]]$variable=="B01001_035",]$estimate) # 25-29
    fracyoungadult <- young/total # fraction in the same house as 1 year ago
    names(fracyoungadult) <- (censusdata[[paste(statename,"b01001")]][censusdata[[paste(statename,"b01001")]]$variable=="B01001_001",]$GEOID)

    drawprop <- function(title, xlab, dat, drawylab=TRUE) {
        plot(x=NA, y=NA, xlim=c(-4,98), ylim=c(0,120), main=title, xlab=xlab, ylab=ifelse(drawylab,"% CBGs in cluster",NA), cex.lab=3.1, cex.main=2.8, cex.axis=2.8, padj=0.3, axes=FALSE)
        axis(1, at=seq(0,100,10), cex.axis=2.7, padj=0.35)
        axis(2, at=seq(0,100,20), cex.axis=2.7)
        binsize <- 10
        bins <- seq(0,100,binsize)/100
        bins[length(bins)] <- 1.001
        for (binnum in 1:(length(bins)-1)) {
            cbgs <- names(dat)[!is.na(dat) & dat>=bins[binnum] & dat<bins[binnum+1] & (as.numeric(names(dat)) %in% as.numeric(data.clusters[["gmm5"]]$ind))]
            y <- 0
            for (cluster in c("A","B","C","D","E")) {
                temp <- sum(data.clusters[["gmm5"]]$cluster[as.numeric(data.clusters[["gmm5"]]$ind) %in% as.numeric(cbgs)]==cluster)
                rect(xleft=bins[binnum]*100, xright=(bins[binnum]+0.92*(bins[binnum+1]-bins[binnum]))*100, ybottom=100*y/length(cbgs), ytop=100*(y+temp)/length(cbgs), col=pal.cluster[cluster], lwd=0.2)
                y <- y+temp
            }
            text(x=100*(bins[binnum]+bins[binnum+1])/2, y=101, lab=length(cbgs), col="gray20", cex=2.3, adj=c(0,0.5), srt=90)
            text(x=1, y=102, lab="N=", col="gray20", cex=2.3, adj=c(1,0))
        }
    }
    png(paste(figuredirectory,"proportions-gmm5-samehouse-",statename,".png",sep=""), width=1600, height=360)
    par(mfrow=c(1,4),
        mar=c(5.2,6,2.8,1.2), #bottom, left, top, and right.
        mgp=c(3.7,0.8,0))
    capstatename <- paste(toupper(substr(statename,1,1)),substr(statename,2,100),sep="")
    drawprop(title=paste("Same house,", capstatename), xlab="% in same house as last year", dat=fracsame)
    drawprop(title=paste("Renter,", capstatename), xlab="% renters", dat=fracrenter, drawylab=FALSE)
    drawprop(title=paste("Undergrad or grad,", capstatename), xlab="% enrolled undergrad or grad", dat=fracgrad, drawylab=FALSE)
    drawprop(title=paste("Young adult,", capstatename), xlab="% ages 18-29y", dat=fracyoungadult, drawylab=FALSE)
    dev.off()

##################
# device count vs destination cbg
# the number of devices that are seen near their homes (or only away from their homes)
# uses raw social distancing data
    dates <- dates.socialdistancing
    dates <- dates[as.Date(dates)>="2020/02/23" & as.Date(dates)<="2020/06/18"] # 117 days of data
    dat <- matrix(data=NA, nrow=length(dates), ncol=length(shapefile.cbg$GEOID))
    colnames(dat) <- shapefile.cbg$GEOID

    for (datenum in 1:length(dates)) {
        d <- dates[datenum]
        print(d)
        for (sdnum in 1:nrow(data.socialdistancing[[paste(statename,d)]])) {
            x <- unlist(fromJSON(gsub('\"\"', '"', data.socialdistancing[[paste(statename,d)]]$destination_cbg[sdnum])))
            if (length(x)>0) {
                if (sum(names(x)==data.socialdistancing[[paste(statename,d)]]$origin_census_block_group[sdnum])==0) {
                    dat[datenum, match(as.numeric(data.socialdistancing[[paste(statename,d)]]$origin_census_block_group[sdnum]), as.numeric(colnames(dat)))] <- 0 # people only seen away from home cbg
                } else {
                    dat[datenum, match(as.numeric(data.socialdistancing[[paste(statename,d)]]$origin_census_block_group[sdnum]), as.numeric(colnames(dat)))] <- x[names(x)==data.socialdistancing[[paste(statename,d)]]$origin_census_block_group[sdnum]]/data.socialdistancing[[paste(statename,d)]]$device_count[sdnum] # fraction seen in home cbg
                }
            }
        }
    }

    png(paste(figuredirectory,"onlyawayfromhome-clusters-",statename,".png",sep=""), width=800, height=1200)
    par(mfrow=c(5,1),
        mar=c(3.7,6.9,2.6,0.5), #bottom, left, top, and right.
        mgp=c(4.6,0.9,0))
    for (cluster in c("A","B","C","D","E")) {
        cbgs <- data.clusters[["gmm5"]]$ind[data.clusters[["gmm5"]]$cluster==cluster]
        ymax <- 60
        plot(x=NA, y=NA, xlim=c(as.Date("2020/02/23")+3,max(as.Date(dates))-3), ylim=c(0,ymax), ylab="% only away", xlab="", main=paste("% only away from home,",paste(toupper(substr(statename,1,1)),substr(statename,2,100),sep=""), "cluster", cluster), cex.lab=3.2, cex.main=3.4, cex.axis=3, type="b", yaxs="i", axes=FALSE)
        for (m in 1:12) {
            lines(x=rep(as.Date(paste("2020",m,"01",sep="/")),2), y=c(-100,ymax), col="gray", lwd=0.7)
        }
        for (y in seq(0,100,20)) {
            abline(a=y, b=0, col="gray", lwd=0.7)
        }
        v <- colnames(dat) %in% cbgs
        polygon(x=c(as.Date(dates),rev(as.Date(dates))), y=100*c((1-sapply(1:length(dates), function(i) {quantile(dat[i,v],0.25, na.rm=TRUE)})),
                                                                 rev((1-sapply(1:length(dates), function(i) {quantile(dat[i,v],0.75, na.rm=TRUE)})))), col=pal.cluster[cluster], lwd=0.5, border="darkgray")
        lines(x=as.Date(dates), y=100*(1-sapply(1:length(dates), function(i) {quantile(dat[i,v],0.5, na.rm=TRUE)})), col="black", lwd=2.5)

        axis(2, cex.axis=2.65, las=2)
        axis(1, at=as.Date(paste("2020",1:12,"01",sep="/")), lab=month.abb, cex.axis=3, padj=0.5)
    }
    dev.off()


########
# street map of Seattle WA with some clusters highlighted    
    if (statename=="washington") {
        library(OpenStreetMap)
        library(ggmap)
        
        mp_osm <- openmap(c(47.545,-122.44),c(47.72,-122.15),type='esri-topo')
        png(paste(figuredirectory,"map-clusters-seattle.png",sep=""), width=1500, height=1350)
        plot(mp_osm)
        pal <- pal.cluster
        temp <- data.clusters[["gmm5"]]$cluster[match(shapefile.cbg$GEOID,data.clusters[["gmm5"]]$ind)]
        plot(st_transform(st_geometry(shapefile.cbg),osm()), add=TRUE,
             border="gray20", lwd=0.32, col=ifelse(temp=="E", rgb(r=0.8,g=0.471,b=0.737,alpha=0.35), ifelse(temp=="D", rgb(r=0.12,g=0.471,b=0.706,alpha=0.35), NA)))
        plot(st_transform(st_geometry(shapefile.cbg),osm()), add=TRUE,
             border=ifelse(temp=="D",pal.cluster["D"],ifelse(temp=="E",pal.cluster["E"],NA)), lwd=1.7, col=NA)
        dev.off()
    }
}


