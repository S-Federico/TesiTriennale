t<-read.csv("C:/Users/fonde/PycharmProjects/Tesi triennale/pd_speech_features.csv")
t2<-t
library(Hmisc)
v<-varclus(as.matrix(t2,similarity="spearman",type="data.matrix"))
plot(v)

Threshold=0.01
a<-cutree(v$hclust,h=1-Threshold)
dfa<-as.data.frame((a))
features=data.frame(Name=rownames(dfa),Id=dfa$'(a)')
library(tidyverse)
write_csv(features,'C:/Users/fonde/PycharmProjects/Tesi triennale/Features clusters/features001.csv')