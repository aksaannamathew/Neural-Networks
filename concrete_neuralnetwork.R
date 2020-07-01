library(readr)
library(neuralnet)
library(nnet)
library(psych)
library(DataExplorer)
library(ggplot2)
library(plyr)

#Importing Dataset
Concrete <- read.csv("C:\\Users\\91755\\Desktop\\Assignment\\11 - Neural Network\\concrete.csv")
attach(Concrete)
head(Concrete)

#EDA and Statistical Analysis
str(Concrete)
summary(Concrete)
colnames(Concrete)

#Graphical Representation
pairs.panels(Concrete)
ggplot(Concrete, aes(x=cement, y=strength))+geom_point() + geom_smooth(method = "lm")
ggplot(Concrete, aes(x=slag, y=strength)) +geom_point() + geom_smooth(method = "lm")
ggplot(Concrete, aes(x=ash, y=strength)) + geom_point() + geom_smooth(method = "lm")
ggplot(Concrete, aes(x=age, y=strength)) + geom_point() + geom_smooth(method = "lm")

#Normalisation
normal <- function(x){
  return((x-min(x))/(max(x)-min(x)))
}
concrete_norm <- as.data.frame(lapply(Concrete, normal))
head(concrete_norm)
pairs.panels(concrete_norm)

#Data Splitting
set.seed(123)
split <- sample(2, nrow(concrete_norm), replace = T, prob = c(0.75, 0.25))
train <- concrete_norm[split==1,]
test <- concrete_norm[split==2,]
head(train)

#Model Building
set.seed(222)
Model_1 <- neuralnet(strength~., data = train)
summary(Model_1)
plot(Model_1, rep = "best")

#Evaluation
set.seed(123)
modelresult_1 <- compute(Model_1, test[,1:8])
modelresult_1
pred_1 <- modelresult_1$net.result
cor(pred_1, test$strength)          #Accuracy = 83.79%

#Since the prediction on strength is in normalised form, to compare need 
concrete_max <- max(Concrete$strength)
concrete_min <- min(Concrete$strength)
denormalise <- function(x, min, max){
  return((min-max)*x+min)
}
Actual_pred <- denormalise(pred_1, concrete_min, concrete_max)
data.frame(head(Actual_pred), head(Concrete$strength))

#Model Building Using 2 Hidden Layers
set.seed(123)
Model_2 <- neuralnet(strength~., data = train, hidden = 2)
summary(Model_2)
plot(Model_2, rep = "best")

#Evaluation
set.seed(222)
modelresult_2 <- compute(Model_2, test[,1:8])
modelresult_2
pred_2 <- modelresult_2$net.result
cor(pred_2, test$strength)        #Accuracy = 92.13%

#Model Building Using 5 Hidden Layers
set.seed(1234)
Model_3 <- neuralnet(strength~., data = train, hidden = 5)
summary(Model_3)
plot(Model_3, rep = "best")

#Evaluation
set.seed(2222)
modelresult_3 <- compute(Model_3, test[,1:8])
modelresult_3
pred_3 <- modelresult_3$net.result
cor(pred_3, test$strength)      #Accuracy = 94.41%
#We can get better accuracy by increasing the hidden layers
