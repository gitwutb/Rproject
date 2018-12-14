rm(list = ls(all=T))
gc()
# library(RODBC)
# library(reshape2)
# library(dplyr)
# library(ggplot2)
# library(RMySQL)
# library(stringr)
# library(e1071) 
# library(tcltk)
# library(lubridate)
library(keras)
library(tensorflow)
library(corrplot)
iris <- read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"), header = FALSE) 
names(iris) <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species")
plot(iris$Petal.Length, 
     iris$Petal.Width, 
     pch=21, bg=c("red","green3","blue")[unclass(iris$Species)], 
     xlab="Petal Length", 
     ylab="Petal Width")
M <- cor(iris[,1:4])
corrplot(M, method="circle")

######数据处理（归一化、随机分配成训练及测试数据）
normalize <- function(x) {
  num <- x - min(x)
  denom <- max(x) - min(x)
  return (num/denom)
}
iris_norm <- as.data.frame(lapply(iris[1:4], normalize))
iris[,5] <- as.numeric(iris[,5]) -1
iris <- as.matrix(iris)
# Set `iris` `dimnames` to `NULL`
dimnames(iris) <- NULL
iris[,1:4] <- normalize(iris[,1:4])
summary(iris)
set.seed(1)
ind <- sample(2, nrow(iris), replace=TRUE, prob=c(0.67, 0.33))
iris.training <- iris[ind==1, 1:4]
iris.test <- iris[ind==2, 1:4]
iris.trainingtarget <- iris[ind==1, 5]
iris.testtarget <- iris[ind==2, 5]
# One hot encode training target values（转换）
iris.trainLabels <- to_categorical(iris.trainingtarget)
iris.testLabels <- to_categorical(iris.testtarget)

######*******模型一*********########
model<-keras_model_sequential()
model%>%layer_dense(units = 8, activation = 'relu', input_shape = c(4))%>% layer_dense(units = 3, activation = 'softmax')
# summary(model)
# get_config(model)
# get_layer(model, index = 1)
# model$layers
# model$inputs
# model$outputs
model%>%compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = 'accuracy')
history <- model %>% fit(iris.training,iris.trainLabels,epochs = 200,batch_size = 5,validation_split = 0.2)
plot(history)
plot(history$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="blue", type="l")
# Plot the model loss of the test data
lines(history$metrics$val_loss, col="green")
legend("topright", c("train","test"), col=c("blue", "green"), lty=c(1,1))
plot(history$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="blue", type="l")
lines(history$metrics$val_acc, col="green")
legend("bottomright", c("train","test"), col=c("blue", "green"), lty=c(1,1))
#####通过predict测试model
classes <- model %>% predict_classes(iris.test, batch_size = 128)
table(iris.testtarget, classes)
# Evaluate on test data and labels(评估损失及精准度)
score <- model %>% evaluate(iris.test, iris.testLabels, batch_size = 128)
print(score)

######*******模型二*********########增加模型层数
model <- keras_model_sequential()
model%>%layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>%layer_dense(units = 5, activation = 'relu') %>% 
  layer_dense(units = 3, activation = 'softmax')
model %>% compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = 'accuracy')
model %>% fit(iris.training, iris.trainLabels,epochs = 200, batch_size = 5,validation_split = 0.2)
score <- model %>% evaluate(iris.test, iris.testLabels, batch_size = 128)
print(score)

######*******模型三*********#####对比训练和测试的趋同性
model <- keras_model_sequential() 
model %>%layer_dense(units = 8, activation = 'relu', input_shape = c(4))%>%layer_dense(units = 5, activation = 'relu') %>% 
  layer_dense(units = 3, activation = 'softmax')
model %>% compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = 'accuracy')
history <- model %>% fit(iris.training,iris.trainLabels,epochs = 200, batch_size = 5,validation_split = 0.2)
# Plot the model loss
plot(history$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="blue", type="l")
lines(history$metrics$val_loss, col="green")
legend("topright", c("train","test"), col=c("blue", "green"), lty=c(1,1))
# Plot the model accuracy
plot(history$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="blue", type="l")
lines(history$metrics$val_acc, col="green")
legend("bottomright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

######*******模型四*********#####增加
model <- keras_model_sequential() 
model %>%layer_dense(units = 28, activation = 'relu', input_shape = c(4)) %>%layer_dense(units = 3, activation = 'softmax')
model %>% compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = 'accuracy')
# Fit the model to the data
model %>% fit(iris.training, iris.trainLabels,epochs = 200, batch_size = 5,validation_split = 0.2)
score <- model %>% evaluate(iris.test, iris.testLabels, batch_size = 128)
print(score)

######*******模型五*********#####
model <- keras_model_sequential() 
model %>%layer_dense(units = 28, activation = 'relu', input_shape = c(4)) %>%layer_dense(units = 3, activation = 'softmax')
model %>% compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = 'accuracy')
history <- model %>% fit(iris.training,iris.trainLabels,epochs = 200,batch_size = 3,validation_split = 0.2)

# Plot the model loss
plot(history$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="blue", type="l")
lines(history$metrics$val_loss, col="green")
legend("topright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

# Plot the model accuracy
plot(history$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="blue", type="l")
lines(history$metrics$val_acc, col="green")
legend("bottomright", c("train","test"), col=c("blue", "green"), lty=c(1,1))



######*******模型六*********#####
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>% 
  layer_dense(units = 3, activation = 'softmax')
# Define an optimizer
sgd <- optimizer_sgd(lr = 0.01)
model %>% compile(optimizer=sgd,loss='categorical_crossentropy',metrics='accuracy')
model %>% fit(iris.training, iris.trainLabels,epochs = 200, batch_size = 5,validation_split = 0.2)
score <- model %>% evaluate(iris.test, iris.testLabels, batch_size = 128)
print(score)


sgd <- optimizer_sgd(lr = 0.01)
model %>% compile(optimizer=sgd,loss='categorical_crossentropy',metrics='accuracy')
history <- model %>% fit(iris.training, iris.trainLabels,epochs = 200, batch_size = 5,validation_split = 0.2)
# Plot the model loss
plot(history$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="blue", type="l")
lines(history$metrics$val_loss, col="green")
legend("topright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

# Plot the model accuracy
plot(history$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="blue", type="l")
lines(history$metrics$val_acc, col="green")
legend("bottomright", c("train","test"), col=c("blue", "green"), lty=c(1,1))


save_model_hdf5(model, "my_model.h5")
model <- load_model_hdf5("my_model.h5")
save_model_weights_hdf5("my_model_weights.h5")
model %>% load_model_weights_hdf5("my_model_weights.h5")

json_string <- model_to_json(model)
model <- model_from_json(json_string)
yaml_string <- model_to_yaml(model)
model <- model_from_yaml(yaml_string)