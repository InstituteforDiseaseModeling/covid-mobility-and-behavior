# get-acs2018data.R
# retrieves 2018 ACS data for the SafeGraph clustering paper

library(dplyr)
library(data.table)
library(tidycensus)

outputdirectory <- ""
census_api_key("") # fill in your Census API Key. Apply for one here: https://api.census.gov/data/key_signup.html

#####################################################
# Download data from US Census
for (region in c("washington","texas","california","georgia")) {
    print(region)
    filename <- paste(outputdirectory,"census-forclustering-",region,".rds",sep="")
    if (!file.exists(filename)) {
        statefips <- ifelse(region=="washington","53",ifelse(region=="california","06",ifelse(region=="georgia","13",ifelse(region=="texas","48",NA))))
        censusdata <- list()
        for (tabname in c("B07201",  # mobility, metropolitan
                          "B07202",  # mobility, micropolitan
                          "B07203",  # mobility, not metropolitan or micropolitan
                          "B01001")) { # age
            if (is.null(censusdata[[paste(region,tolower(tabname))]])) {
                bsave <- TRUE
                censusdata[[paste(region,tolower(tabname))]] <- (get_acs(geography = "block group", 
                                                                         table = tabname, 
                                                                         state     = statefips,
                                                                         year      = 2018) %>% as.data.table())
            }
        }
        for (varname in c("B19013_001", # median income
                          "B14002_001", # enrolled in school, total
                          "B14002_019", # enrolled in undergraduate, male
                          "B14002_022", # enrolled in grad school, male
                          "B14002_043", # enrolled in undergraduate, female
                          "B14002_046", # enrolled in grad school, female
                          "B25008_001", # population in occupied housing units
                          "B25008_002", # owner occupied
                          "B25008_003")) { # renter occupied
            if (is.null(censusdata[[paste(region,tolower(varname))]])) {
                bsave <- TRUE
                censusdata[[paste(region,tolower(varname))]] <- (get_acs(geography = "block group", 
                                                                         variables = varname,
                                                                         state     = statefips,
                                                                         year      = 2018) %>% as.data.table())
            }
        }
        if (bsave) {
            saveRDS(censusdata, file=filename)
        }
    }
}
        
if (FALSE) {
for (region in c("washington","texas","california","georgia")) {
    filename <- paste("acs2018-",region,".csv",sep="")
    if (!file.exists(filename)) {
        censusdata <- readRDS(file=paste("census-forclustering-",region,".rds",sep=""))
        pop.acs2018 <- censusdata[[paste(region,"b01001")]][censusdata[[paste(region,"b01001")]]$variable=="B01001_001",]$estimate
        names(pop.acs2018) <- censusdata[[paste(region,"b01001")]][censusdata[[paste(region,"b01001")]]$variable=="B01001_001",]$GEOID
        income <- censusdata[[paste(region,"b19013e1")]]$estimate # income
        names(income) <- censusdata[[paste(region,"b19013e1")]]$GEOID
    
        fracsamehouse <- ((censusdata[[paste(region,"b07201")]][censusdata[[paste(region,"b07201")]]$variable=="B07201_002",]$estimate) +
                          (censusdata[[paste(region,"b07202")]][censusdata[[paste(region,"b07202")]]$variable=="B07202_002",]$estimate) +
                          (censusdata[[paste(region,"b07203")]][censusdata[[paste(region,"b07203")]]$variable=="B07203_002",]$estimate)) /  # same house 1 year ago
            ((censusdata[[paste(region,"b07201")]][censusdata[[paste(region,"b07201")]]$variable=="B07201_001",]$estimate) +
             (censusdata[[paste(region,"b07202")]][censusdata[[paste(region,"b07202")]]$variable=="B07202_001",]$estimate) +
             (censusdata[[paste(region,"b07203")]][censusdata[[paste(region,"b07203")]]$variable=="B07203_001",]$estimate)) # fraction in the same house as 1 year ago
        names(fracsamehouse) <- (censusdata[[paste(region,"b07201")]][censusdata[[paste(region,"b07201")]]$variable=="B07201_001",]$GEOID)
        
        fracrenter <- (censusdata[[paste(region,"b25008_003")]]$estimate/censusdata[[paste(region,"b25008_001")]]$estimate) # frac renter
        names(fracrenter) <- censusdata[[paste(region,"b25008_001")]]$GEOID
        
        grad <- (censusdata[[paste(region,"b14002_019")]]$estimate + censusdata[[paste(region,"b14002_043")]]$estimate + censusdata[[paste(region,"b14002_022")]]$estimate + censusdata[[paste(region,"b14002_046")]]$estimate) # enrolled in undergrad or grad (M+F)
        names(grad) <- censusdata[[paste(region,"b14002_019")]]$GEOID
        fracgrad <- grad/pop.acs2018[match(names(grad), names(pop.acs2018))]
        
        summary(names(income)==names(pop.acs2018)) # sanity check
        summary(names(income)==names(fracgrad)) # sanity check
        temp <- data.frame(cbg=names(income),pop2018=pop.acs2018,medianincome=income, fracsamehouse=round(fracsamehouse,4), fracrenter=round(fracrenter,4), fraccollege=round(fracgrad,4))
        
        write.csv(temp, file=paste("acs2018-",region,".csv",sep=""), row.names=FALSE, quote=FALSE)
    }
}
}
