library(ggplot2)
library(reshape2)
library(ggstatsplot)
library(tcltk)
library(rstatix)
library(PMCMRplus)
library(dplyr)
library(scales)

data <- read.csv(file = '/home/thais/Dev/TVMBench/out.csv', sep = ',', header = T)
data$approach <- factor(data$approach, levels = c("TGC-ES", "TVM-ES", "TGC", "TVM"))

ggplot(data=data, aes(x=model_name, y=tuning_mean, fill=approach))+
  geom_bar(stat="identity", color="black", position=position_dodge(), alpha=.9)+
  geom_errorbar(aes(ymin=tuning_mean-tuning_std, ymax=tuning_mean+tuning_std), width=.2,
                position=position_dodge(.9))+
  scale_fill_brewer(palette="Greys")+
  theme(axis.text.x=element_text(angle=45,hjust=.5,vjust=0.5))+
  #theme(legend.justification=c(1, 1),legend.position=c(.9, .9),legend.title=element_blank())+
  theme(panel.background = element_rect(fill = 'white', colour = 'gray'),
        panel.grid.major = element_line(color = 'light gray'),
        panel.grid.minor = element_line(color = 'light gray'),
        axis.text.y = element_text(size=8),
        legend.background = element_rect(fill=alpha('white', 0.6)))+
  labs(x="Deep Learning Model", y="Normalized Tuning Time")+
  scale_fill_manual(values = c("#E07B39", "#4A4A4A", "#5C9E68", "#6194C2FF"))+ 
  #geom_hline(yintercept=1, linetype="dashed", size=.5, color = "black")+
  theme(legend.position="top", legend.title=element_blank())
