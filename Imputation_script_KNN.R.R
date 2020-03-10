library(tidyverse)
library(ggplot2)
library(bnstruct)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


df = read.csv("Data/Processed_data.csv")
df = df[,-1]

df = df[which(rowMeans(!is.na(df)) > 0.5), ]
df = as.matrix(df)

cat_cols = c('birth_attendant','birth_place','cigs_before_preg','birth_mn','birth_dy','f_education','f_hispanic','f_race6','gonorrhea','labour_induced','m_nativity','m_education','m_hispanic','admit_icu','m_race6','m_transferred','infections','m_morbidity','riskf','payment','mn_prenatalcare_began','num_prenatal_visits','prior_births_dead','prior_births_living','prior_terminations','delivery_method','res_status','prev_cesarean','num_prev_cesareans','infant_sex')

cat_idx = match(cat_cols, names(df))
cat_idx = as.vector(cat_idx)

df_imp = knn.impute(df, cat.var = cat_idx)

df_imp = as.data.frame(df_imp)

df_imp['weight_change'] = df_imp['m_deliveryweight'] - df_imp['prepreg_weight']

df_imp_bin = df_imp
df_imp_bin$birthweight_bin = ifelse(df_imp_bin$birthweight_g < 2500, 1, 0)

df_imp_cat = df_imp_bin
df_imp_cat$birthweight_cat = ifelse(df_imp_cat$birthweight_g < 1500, 2,
                                    df_imp_cat$birthweight_bin)

drops_bin <- "birthweight_g"
drops_cat <- c("birthweight_g", "birthweight_cat")

df_imp_bin = df_imp_bin[ , !(names(df_imp_bin) %in% drops_bin)]
df_imp_cat = df_imp_cat[ , !(names(df_imp_cat) %in% drops_cat)]

write.csv(df_imp, file="Data/KNN_data.csv")
write.csv(df_imp_bin, file="Data/KNN_data_bin.csv")
write.csv(df_imp_cat, file="Data/KNN_data_cat.csv")
