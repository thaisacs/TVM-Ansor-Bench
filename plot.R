library(ggplot2)
library(reshape2)
library(ggstatsplot)
library(tcltk)
library(rstatix)
library(PMCMRplus)

data <- read.csv(file = '/home/thais/Dev/TVMBench/dataset.csv', sep = ',', header = T)

ggplot(data=data, aes(x=iteration, y=value, group=tipo)) +
  #geom_errorbar(aes(ymin=value-desvio, ymax=value+desvio, color=tipo), width=0.2, size=.2) +
  geom_line(aes(color=tipo), size=1) +
  #coord_cartesian(ylim=c(0, 3.7), xlim=c(1,1000))+
  geom_vline(xintercept=0, linetype="dashed", size=1)+
  #labs(color = element_blank(), x="Iteração", y="Média Normalizada do Melhor Tempo")+
  #theme(legend.justification=c(1, 1),legend.position=c(.9, .9),legend.title=element_blank())+
  #theme(panel.background = element_rect(fill = 'white', colour = 'gray'),
  #      panel.grid.major = element_line(color = 'light gray'),
  #      panel.grid.minor = element_line(color = 'light gray'),
  #      axis.title.y=element_blank(),
  #      axis.text.y = element_text(size=8),
  #      legend.background = element_rect(fill=alpha('white', 0.6)))+
  #ggtitle('a) Melhor Tempo de Execucao Encontrado')+
  scale_x_continuous(breaks = seq(0, 1000, 50))


ggplot(data=data, aes(x=iteration, y=value, group=tipo)) +
  geom_errorbar(aes(ymin=value-desvio, ymax=value+desvio, color=tipo), width=0.2, size=.2) +
  geom_line(aes(color=tipo), size=1) +
  #coord_cartesian(ylim=c(1, 2), xlim=c(1,1000))+
  geom_vline(xintercept=0, linetype="dashed", size=1)+
  labs(color = element_blank(), x="Iteração", y="Média Normalizada do Melhor Tempo")+
  theme(legend.justification=c(1, 1),legend.position=c(.9, .9),legend.title=element_blank())+
  #theme(panel.background = element_rect(fill = 'white', colour = 'gray'),
  #      panel.grid.major = element_line(color = 'light gray'),
  #      panel.grid.minor = element_line(color = 'light gray'),
  #      axis.title.y=element_blank(),
  #      axis.text.y = element_text(size=8),
  #      legend.background = element_rect(fill=alpha('white', 0.6)))+
  ggtitle('a) Melhor Tempo de Execucao Encontrado')+
  scale_x_continuous(breaks = seq(0, 1000, 50))
 