# Sample code to build models in the main analysis
# You can just run this script and it will call the python code directly
# try from the command line: R CMD BATCH sample_model_building.r
# or download Rstudio (www.rstudio.org) and run from within the IDE

# important libraries
require(data.table)
require(caret)
require(pROC)
require(Hmisc)
require(glmnet)
require(randomForest)
require(gbm)

# Let's build a sample data set
# The actual data used in this investigation can be accessed
# for free after signing a DUA at:
# https://mimic.physionet.org/

# number of observations
set.seed(20172017)
N <- 10000

# the outcome, death or ICU LOS >= 7 days
Y <- sample(0:1, size = N, replace = TRUE, prob = c(0.8,0.2))

# Let's make 3 variables that are somewhat correlated with the outcome 
# and introduce some additional bias
age <- 50 + Y * 10 + rnorm(N, 3, sd = 4)
sbp <- 110 - Y * 10 + rnorm(N, 8, sd = 3)
any_vent <- ifelse(Y * runif(N, 0, 1) >= 0.5, 1, 0)

# and randomly corrupt a few of these to make model fitting a bit more challenging
age_crpt_indx <- sample(1:N, size = round(N * 0.3), replace = F)
age[age_crpt_indx] <- sqrt(age[age_crpt_indx]) * 5 + 10 * cos(age[age_crpt_indx])
sbp_crpt_indx <- sample(1:N, size = round(N * 0.4), replace = F)
sbp[sbp_crpt_indx] <- 20 + log(sbp[sbp_crpt_indx]) * 15 + 5 * sin(sbp[sbp_crpt_indx])

# Let's make 5 more variables that are essentially random
# i.e. uncorrelated with the outcome
gender <- sample(c('m','f'), size = N, replace = T)
pao2 <- rnorm(N, 120, 20)
temp <- rnorm(N, 37, 2)
any_cardiac_arrest <- sample(0:1, size = N, replace = T, prob = c(0.05,0.95))
creatinine <- rnorm(N, 1.5 , 0.2) 

# Now let's make some sample unstructured data
note_template_terms <- ('This patient is a 53 year old female with a history 
                        of hypertension, bipolar disorder, chronic kidney 
                        disease and asthma who was admitted to the intensive 
                        care unit for acute respiratory failure. Her course 
                        has been complicated by sepsis and delirium and she 
                        remains on broad-spectrum antibiotics. A plain chest 
                        radiograph reveals an evolving infiltrate at the right 
                        lung base, and her white blood cell count continues to 
                        rise. Current drips include heparin, norepinephrine, 
                        and vasopressin.')

# now let's scramble those words to make unique combinations
# the order, although non-sensical, won't matter for this sample analysis
note_tokens <- unlist(strsplit(note_template_terms,' '))
note_tokens <- gsub('\n', '', note_tokens) # remove newlines
note_tokens <- grep('^$', note_tokens, value = TRUE, invert = TRUE) # remove empty
sample_notes <- sapply(1:N, function(x) {
    wcount <- round(rnorm(1,150, 20))
    token_indices <- sample(1:length(note_tokens), size = wcount, replace = T)
    paste0(note_tokens[token_indices], collapse = ' ')
})

# now add in select text terms correlated with the outcome
sample_notes[Y == 1] <- paste(sample_notes[Y == 1] , 
                              sample(c('poor prognosis', 'very sick', ''),
                                     size=sum(Y),replace=TRUE,prob = c(0.8,0.15,0.05)))
sample_notes[Y == 0] <- paste(sample_notes[Y == 0], 
                              sample(c('feeling well', ''),sum(Y==0),replace=TRUE))

# create a data frame
df <- data.table(pt_id = 1:N, age, sbp, any_vent, gender, pao2, temp,
                        any_cardiac_arrest, creatinine, Y)

# export the text data for analysis
dir.create('corpus_txt/', showWarnings = FALSE)
invisible(sapply(1:N, function(n) {
  writeLines(sample_notes[n], paste0('corpus_txt/',n,'.txt'))
}))

# Now run the Python code
# NB. this can take 10 minutes or more
system('python3 sample_text_analysis_main.py')

# now import the text output of the Python code
df_ngrams <- fread('ngrams_dtm.csv')
df_key_terms <- fread('text_results_keyterms.csv')

# merge text features together and ignore duplicates
key_terms_unique <- names(df_key_terms)[! names(df_key_terms) %in% names(df_ngrams)]
df_text_terms <- merge(df_ngrams, 
                       df_key_terms[,c('pt_id',key_terms_unique),with=F], 
                       by = 'pt_id')

# fix variable names
names(df_text_terms)[-1] <- paste0('term_',
                              gsub(' ','_', 
                                   names(df_text_terms)[-1]))

# transform word counts per document to binary presence/absence
df_text_terms[,grep("term_",colnames(df_text_terms), value=TRUE) := 
                sapply(.SD, function(x) x > 0),
                                             .SDcols = grep('term_',names(df_text_terms))]

# transform word counts to yes/no presence for each document
df_text_terms_txf <- data.table(pt_id = df_text_terms$pt_id,
                                df_text_terms[,-1,with=F] > 0)

# set up training and testing data stratified on the outcome
training <- createDataPartition(df$Y, p=0.75, list=FALSE)[,1]
testing <- (1:nrow(df))[-training]

# do some variable selection on the text data
x <- df_text_terms_txf[pt_id %in% df[training]$pt_id] # just using TRAINING DATA!!!
x <- x[order(pt_id)]
y <- df[training]$Y
x_ptids <- x$pt_id
x <- x[, pt_id := NULL]
# also remove all columns already present in the data set
keep_cols <- which(! names(x) %in% names(df))
x <- x[, keep_cols, with = FALSE]

# now do variable selection
term_select <- cv.glmnet(as.matrix(x), y, family = 'binomial', parallel = FALSE)
# use cv.glmnet for CV in the future...
# then get the coefficients using: 
terms_coeff <- coef(term_select, s = "lambda.min")[,1] # FOR CV ONLY
# keep up to the top 20 non-zero
terms_coeff_filt <- na.exclude(names(sort(abs(terms_coeff[terms_coeff != 0 & names(terms_coeff) != '(Intercept)']), 
                               decreasing = T))[1:20])

# now get variable name list and merge with dt.train
add_terms <- df_text_terms_txf[, c('pt_id',terms_coeff_filt),with=F] 
# merge text data into full data set
df <- merge(df, add_terms, by = 'pt_id')

# define limited variable set of structured data
var_names_struct <- grep('term_', names(df), value = TRUE, invert = TRUE)

# validation details
## 5-times repeated 10-fold repeated CV
fitControl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5,
  verboseIter = TRUE)

# set tuning grid for each model type for which it's needed
rfGrid <- data.frame(mtry = 5:15)
gbmGrid <- expand.grid(n.trees = c(100, 250, 500),
                       interaction.depth = c(5,10,15,20),
                       shrinkage = c(0.005, 0.01),
                       n.minobsinnode = c(5,10,15))
enetGrid <- expand.grid(alpha = seq(0.1,1,by=0.1),
                        lambda = c(0.001, 0.003, 0.005, 0.010, 0.015,
                                   0.020, 0.025, 0.030))

# Running the follow models may take a while (a few hours)
# To speed things up, uncomment the following lines to build
# multiple models in parallel
#require(doMC)
#registerDoMC(6)

# Now train a model using structured and structured + unstructured data for each type
# Logistic regression
lr.str <- train(as.numeric(Y) ~ ., 
                method = 'glm',
                family = 'binomial', 
                trControl = fitControl,
                data = df[training,var_names_struct,with=F])
lr.unstr <- train(as.numeric(Y) ~ ., 
                method = 'glm',
                family = 'binomial', 
                trControl = fitControl,
                data = df[training])

# Penalized logistic regression (elastic net)
en.str <- train(as.numeric(Y) ~ ., 
                method = 'glmnet',
                tuneGrid = enetGrid,
                trControl = fitControl,
                data = df[training,var_names_struct,with=F])
en.unstr <- train(as.numeric(Y) ~ ., 
                  method = 'glmnet',
                  tuneGrid = enetGrid,
                  trControl = fitControl,
                  data = df[training])

# Random forests
rf.str <- train(as.numeric(Y) ~ ., 
                method = 'rf',
                tuneGrid = rfGrid,
                trControl = fitControl,
                data = df[training,var_names_struct,with=F])
rf.unstr <- train(as.numeric(Y) ~ ., 
                  method = 'rf',
                  tuneGrid = rfGrid,
                  trControl = fitControl,
                  data = df[training])

# Gradient boosted machines
gbm.str <- train(as.numeric(Y) ~ ., 
                method = 'gbm',
                tuneGrid = gbmGrid,
                trControl = fitControl,
                data = df[training,var_names_struct,with=F])
gbm.unstr <- train(as.numeric(Y) ~ ., 
                  method = 'gbm',
                  tuneGrid = gbmGrid,
                  trControl = fitControl,
                  data = df[training])

# generate predictions on hold-out test set

# a helper function to make predicted probabilities in useful range
fix_range <- function(vals, low = 0, high = 1) {
  vals [vals < low] <- low
  vals [vals > high] <- high
  vals
}

## structured data models
lr.str.preds <- fix_range(predict(lr.str, df[testing]))
en.str.preds <- fix_range(predict(en.str, df[testing]))
rf.str.preds <- fix_range(predict(rf.str, df[testing]))
gbm.str.preds <- fix_range(predict(gbm.str, df[testing]))

## structured + unstructured data models
lr.unstr.preds <- fix_range(predict(lr.unstr, df[testing]))
en.unstr.preds <- fix_range(predict(en.unstr, df[testing]))
rf.unstr.preds <- fix_range(predict(rf.unstr, df[testing]))
gbm.unstr.preds <- fix_range(predict(gbm.unstr, df[testing]))

# generate ROC data for each model
## structured data models
lr.str.roc <- roc(predictor = lr.str.preds, response = df[testing]$Y, ci = TRUE)
en.str.roc <- roc(predictor = en.str.preds, response = df[testing]$Y, ci = TRUE)
rf.str.roc <- roc(predictor = rf.str.preds, response = df[testing]$Y, ci = TRUE)
gbm.str.roc <- roc(predictor = gbm.str.preds, response = df[testing]$Y, ci = TRUE)

## structured + unstructured data models
lr.unstr.roc <- roc(predictor = lr.unstr.preds, response = df[testing]$Y, ci = TRUE)
en.unstr.roc <- roc(predictor = en.unstr.preds, response = df[testing]$Y, ci = TRUE)
rf.unstr.roc <- roc(predictor = rf.unstr.preds, response = df[testing]$Y, ci = TRUE)
gbm.unstr.roc <- roc(predictor = gbm.unstr.preds, response = df[testing]$Y, ci = TRUE)

# make two-panel roc plot
par(mfrow = c(1,2))
plot_cols <- c('red', 'blue','forest green', 'dark orange')

# Plot structured group
plot(lr.str.roc, print.auc = TRUE, print.auc.cex = 0.8, col = plot_cols[1], 
     main = 'Discrimination of models using structured data')
plot(en.str.roc, col = plot_cols[2], add = TRUE, print.auc = TRUE, print.auc.cex = 0.8, print.auc.y = 0.45)
plot(rf.str.roc, col = plot_cols[3], add = TRUE, print.auc = TRUE, print.auc.cex = 0.8, print.auc.y = 0.40)
plot(gbm.str.roc, col = plot_cols[4], add = TRUE, print.auc = TRUE, print.auc.cex = 0.8, print.auc.y = 0.35)
legend('bottomright', legend = c('LR', 'EN', 'RF', 'GBM'), bty = 'n',
       col = plot_cols, lwd = 2, cex = .8, y.intersp = 0.5, seg.len = 1)

# plot structured + unstructured group
plot(lr.unstr.roc, print.auc = TRUE, print.auc.cex = 0.8, col = plot_cols[1], 
     main = 'Discrimination of models using structured and \nunstructured data')
plot(en.unstr.roc, col = plot_cols[2], add = TRUE, print.auc = TRUE, print.auc.cex = 0.8, print.auc.y = 0.45)
plot(rf.unstr.roc, col = plot_cols[3], add = TRUE, print.auc = TRUE, print.auc.cex = 0.8, print.auc.y = 0.40)
plot(gbm.unstr.roc, col = plot_cols[4], add = TRUE, print.auc = TRUE, print.auc.cex = 0.8, print.auc.y = 0.35)
legend('bottomright', legend = c('LR', 'EN', 'RF', 'GBM'), bty = 'n',
       col = plot_cols, lwd = 2, cex = .8, y.intersp = 0.5, seg.len = 1)

