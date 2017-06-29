require(data.table)
setwd("~/Lectures/COS424/homework2/imputation")

# import data
d <- fread("../data/background.csv")
train <- fread("../data/train.csv")

setkeyv(d, "challengeID")
setkeyv(train, "challengeID")

# remove IDs
noIds <- c("idnum", "mothid4", "fathid4", "fathid3", "mothid3", "fathid2", "mothid2", "fathid1", "mothid1")
d[, (noIds):=NULL]

# remove age and date columns
# dateValues <- c("date", "year", "yrs", "yr", "mon")
# ageDate <- unique(unlist(lapply(dateValues, function(x) {colnames(d)[grepl(x, colnames(d))]})))
# d[, (ageDate):=NULL]

# remove character columns
charCols <- colnames(d)[sapply(d, function(x) {class(x) == "character"})]
d[, (charCols) := NULL]

# create extra columns for skips
skip <- data.table(d == -6)
## Rename the columns of the missing indicators to start with miss_
colnames(skip) <- paste("skip_", colnames(d), sep = "")
## Keep only the columns where some are missing
msk <- apply(skip, 2, var, na.rm = T) > 0
skip <- skip[, msk, with = F]
skip$challengeID <- d$challengeID
# merge them back together
d <- merge(d, skip, all=TRUE)

# create extra columns for refuse
refuse <- data.table(d == -1)
## Rename the columns of the missing indicators to start with miss_
colnames(refuse) <- paste("refuse_", colnames(d), sep = "")
## Keep only the columns where some are missing
msk <- apply(refuse, 2, var, na.rm = T) > 0
refuse <- refuse[, msk, with = F]
refuse$challengeID <- d$challengeID
# merge them back together
d <- merge(d, refuse, all=TRUE)


# convert NA codes into real NA
d[d < 0] <- NA
d$hv5_ppvtpr[d$hv5_ppvtpr == "Other"] <- NA
d$hv5_wj9pr[d$hv5_wj9pr == "Other"] <- NA
d$hv5_wj10pr[d$hv5_wj10pr == "Other"] <- NA

# remove all-NA columns
allNa <- colnames(d)[unlist(lapply(d, function(x) all(is.na(x))))]
d[, (allNa):=NULL]

filled.data <- lapply(
  d, function(x) {
    replacement <- 0
    if(class(x) == "numeric") {
      ## Find the mean for numeric columns
      replacement <- mean(na.omit(x[x>0]))
    } else {
      ## Identify the unique values of that variable
      ux <- unique(na.omit(x[x > 0]))  
      ## Find the mode
      replacement <- ux[which.max(tabulate(match(na.omit(x[x > 0]), ux)))]
    }
    if (is.na(replacement)) replacement <- 1
    ## Replace with the mode if missing
    x[is.na(x)] <- replacement
    return(x)
  }
)
d <- as.data.table(filled.data)
rm(filled.data, refuse, skip)

# remove columns that contain only one value
singleValueCols <- sapply(d, function(x) {length(table(x)) == 1})
d[, (colnames(d)[singleValueCols]):=NULL]

# remove highly correlated columns
# numerics <- lapply(d, class) != "character"
cm <- cor(d, method = "spearman")
cm[upper.tri(cm)] <- 0
diag(cm) <- 0
corTooLarge <- apply(cm, 2, function(x) any(abs(x) > 0.6))
d[, (names(corTooLarge[corTooLarge == TRUE])):=NULL]
rm(cm)

write.csv(d, file = "imputed-large.csv", row.names = F)
