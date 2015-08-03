# ebay_ipad_kaggle.R
# EdX - Analytics Edge - Unit 6
# 2015-07-28

# SETTINGS AND LIBRARIES ####
setwd('/Users/r2/MOOC/Analytics Edge - MITx/Kaggle')
library(dplyr)
library(caret)
library(randomForest)
library(tm)

# GET AND CLEAN DATA ####

# Load data
rawTrain <- tbl_df(read.csv('eBayiPadTrain.csv', stringsAsFactors = F))
rawTest <- tbl_df(read.csv('eBayiPadTest.csv', stringsAsFactors = F))

# Put description as last variable to make data easier to look at
train <- select(rawTrain, 2:11, 1)
test <- select(rawTest, 2:10, 1)

# Convert suitable variables to factors
train <- mutate(train,
                sold = as.factor(train$sold),
                biddable = as.factor(train$biddable),
                condition = as.factor(train$condition),
                cellular = as.factor(train$cellular),
                carrier = as.factor(train$carrier),
                color = as.factor(train$color),
                storage = as.factor(train$storage),
                productline = as.factor(train$productline)
                 )

test <- mutate(test,
                biddable = as.factor(test$biddable),
                condition = as.factor(test$condition),
                cellular = as.factor(test$cellular),
                carrier = as.factor(test$carrier),
                color = as.factor(test$color),
                storage = as.factor(test$storage),
                productline = as.factor(test$productline)
                )

# BUILD NEW FEATURES ####

# Add feature for how many characters listing's description
train <- mutate(train, nchar = nchar(train$description))
test <- mutate(test, nchar = nchar(test$description))

# Add logical feature if the listing has a description or not
train <- mutate(train, hasDescription = ifelse(train$nchar == 0, 0, 1))
test <- mutate(test, hasDescription = ifelse(test$nchar == 0, 0, 1))

# CORPUS FROM TEXTUAL DATA ####

# Create corpus by combining test and training sets
trainDescription <- as.matrix(train$description)
testDescription <- as.matrix(test$description)
corpus <- Corpus(VectorSource(c(trainDescription, testDescription)))

# Pre-process text
corpus <-  tm_map(corpus, content_transformer(tolower)) ; corpus <-  tm_map(corpus, PlainTextDocument)
corpus <-  tm_map(corpus, removePunctuation)
corpus <-  tm_map(corpus, removeWords, stopwords('english'))
corpus <-  tm_map(corpus, stemDocument)

# Build a document term matrix with 0.99 sparsity
dtm <- DocumentTermMatrix(corpus)
sparseDtm <- removeSparseTerms(dtm, 0.99)
words <- as.matrix(sparseDtm)
rownames(words) <- NULL
words <- as.data.frame(words)
colnames(words) <- make.names(colnames(words))

# Divide bag to train and test set
trainWords <- head(words, nrow(train))
testWords <- tail(words, nrow(test))

# CHOOSE VARIABLES TO USE IN THE MODEL ####
trainMod <- select(train, -description)
trainMod <- cbind(trainMod, trainWords)

testMod <- select(test, -description)
testMod <- cbind(testMod, testWords)

# BUILD MODELS ####
trainMod <- as.data.frame(trainMod)
testMod <- as.data.frame(testMod)

# General linear model
glmMod <- glm(sold ~ ., data = trainMod, family = binomial)
glmPredictTrain <- predict(glmMod, newdata = trainMod, type = 'response')
table(train$sold, glmPredict > 0.5)

glmStep <- step(glmMod)
glmPredictStepTrain <- predict(glmStep, newdata = trainMod, type = 'response')
table(train$sold, glmPredictStepTrain > 0.5)

# Random forest model
rfMod <- train(sold ~ ., data = trainMod, method = 'rf')
rfPredictTrain <- predict(rfMod, newdata = trainMod)
table(train$sold, rfPredict)

# Final predictions on test set
glmPredict <- predict(glmMod, newdata = testMod, type = 'response')
glmPredictStep <- predict(glmStep, newdata = testMod, type = 'response')

rfPredict <- predict(rfMod, newdata = testMod, type = 'prob')


# CREATE SUBMISSION FILE
glmSubmission = data.frame(UniqueID = test$UniqueID, Probability1 = glmPredictStep)
write.csv(glmSubmission, "GLMSubmissionDescriptionLog.csv", row.names=FALSE)

rfSubmission = data.frame(UniqueID = test$UniqueID, Probability1 = rfPredict)
write.csv(rfSubmission, "SubmissionDescriptionLog.csv", row.names=FALSE)




