#Installing libraries
library(readr)
library(neuralnet)
library(nnet)
library(DataExplorer)
library(ggplot2)
library(psych)
library(plyr)
library(caret)

#Importing Dataset
Forestfires <- read.csv("C:\\Users\\91755\\Desktop\\Assignment\\11 - Neural Network\\forestfires.csv")
attach(Forestfires)
head(Forestfires)

#EDA and Statistical Analysis
sum(is.na(Forestfires))
Forestfires <- Forestfires[c("month", "day", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain", "area", "size_category")]
str(Forestfires)
head(Forestfires)
summary(Forestfires)

table(Forestfires$month)
Forestfires$month <- as.integer(factor(Forestfires$month, levels = c("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
                                       labels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)))
table(Forestfires$day)
Forestfires$day <- as.integer(factor(Forestfires$day, levels = c("mon", "tue", "wed", "thu", "fri", "sat", "sun"),
                                     labels = c(1, 2, 3, 4, 5, 6, 7)))
table(Forestfires$size_category)
size_category <- as.numeric(revalue(size_category, c("large"="1", "small"="2")))

#Graphical Representation
plot_correlation(Forestfires)
pairs.panels(Forestfires)
ggplot(Forestfires,mapping = aes(x=temp,y=area))+geom_point()+geom_smooth(method="lm")
ggplot(Forestfires,mapping = aes(x=wind,y=area))+geom_point()+geom_smooth(method="lm")

#Normalization
normal <- function(x)
{
  return((x-min(x))/(max(x)-min(x)))
}
fire_norm <- as.data.frame(lapply(Forestfires[,-12], FUN=normal))
Forest_fire <- cbind(fire_norm, size_category)
as.data.frame(Forest_fire)
head(Forest_fire)
summary(Forest_fire)

#Data Splitting
set.seed(123)
split <- createDataPartition(Forest_fire$size_category, list = F, p=0.75)
train <- Forest_fire[split,]
test <- Forest_fire[-split,]
head(train)

#Model Building
model_1 <- neuralnet(size_category~., data = train)
summary(model_1)
plot(model_1, rep = "best")

#Evaluation
set.seed(123)
modelresult_1 <- compute(model_1, test[,1:11])
modelresult_1
pred_1 <- modelresult_1$net.result
cor(pred_1, test$size_category)        #Accuracy = 96.7%

data.frame(head(pred_1), head(Forest_fire$size_category))

#Since the prediction on profit is normalised
#need to denormalised to get the actual predicted profit 
size_max <- max(Forest_fire$size_category)
size_min <- min(Forest_fire$size_category)

denormalized <- function(x, min, max){
  return((max-min)*x + min)
}
predprofit <- denormalized(pred_1, size_min, size_max)
data.frame(head(Forest_fire$size_category), head(predprofit))

#Model Building Using 2 Hidden Layers
set.seed(222)
model_2 <- neuralnet(size_category~.,data = train, hidden = 2)
plot(model_2, rep = "best")

#Evaluation
set.seed(123)
modelresult_2 <- compute(model_2, test[,1:11])
modelresult_2
pred_2 <- modelresult_2$net.result
cor(pred_2, test$size_category)    #Accuracy = 97.62%

#Model Building Using 5 Hidden Layers
set.seed(2222)
model_3 <- neuralnet(size_category~., data = train, hidden = 5)
plot(model_3, rep = "best")

#Evaluation
set.seed(1234)
modelresult_3 <- compute(model_3, test[,1:11])
modelresult_3
pred_3 <- modelresult_3$net.result
cor(pred_3, test$size_category)       #Accuracy = 91.11%
# we get better accuracy 97% when  we increase the hidden layer.