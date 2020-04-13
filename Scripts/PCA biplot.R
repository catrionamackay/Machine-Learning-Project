rm(list=ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(ggplot2)
library(ggfortify)
library(tidyverse)
library(ggthemes)

df = read.csv('Data/Pipeline_data_toy_bin.csv')
df = df[, -1]
X = df[, -40]
y = df[, 40]

pca <- prcomp(X, scale. = TRUE, center = TRUE)

for_plot <- as_tibble(X) %>% add_column(LBW = y)
for_plot['Class'] <- ifelse(y == 1, 'Low birthweight', 'Normal birthweight')

autoplot(pca, loadings = TRUE, loadings.label = TRUE, data = for_plot,
         colour = 'Class', alpha = 0.2, loadings.colour = 'black',
         loadings.label.colour = 'black')
