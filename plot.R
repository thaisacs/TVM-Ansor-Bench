library(ggplot2)
library(reshape2)
library(ggstatsplot)
library(tcltk)
library(rstatix)
library(PMCMRplus)
library(dplyr)
library(scales)

data <- read.csv(file = '/home/thais/Dev/TVMBench/csvs/best.csv', sep = ',', header = T)
ggplot(data=data, aes(x=iteration, y=value, group=tipo, ymin=value-(2*desvio), ymax=value+(2*desvio), fill=tipo, color=tipo)) +
  geom_ribbon(alpha=.3, lty=0) +
  geom_line(size=1.5) +
  coord_cartesian(ylim=c(1, 1.3), xlim=c(1,1000))+
  geom_hline(yintercept=1.029, linetype="dashed", size=.5)+
  geom_vline(xintercept=326, linetype="dashed", size=.5)+
  theme(legend.justification=c(1, 1),legend.position=c(.9, .9),legend.title=element_blank())+
  theme(panel.background = element_rect(fill = 'white', colour = 'gray'),
        panel.grid.major = element_line(color = 'light gray'),
        panel.grid.minor = element_line(color = 'light gray'),
        axis.text.y = element_text(size=8),
        legend.background = element_rect(fill=alpha('white', 0.6)))+
  #ggtitle('a) The best solution found in each search iteration (Error ribbons indicate ± 2 SEM)')+
  ggtitle('       Error ribbons indicate ± 2 SEM')+
  labs(x="Iteration", y="Normalized Execution Time")+
  scale_x_continuous(breaks = seq(0, 1000, 100))+
  scale_color_manual(values=c('#E07B39', '#6194C2FF'), aesthetics = c("colour", "fill"))

data <- read.csv(file = '/home/thais/Dev/TVMBench/csvs/acc.csv', sep = ',', header = T)
ggplot(data=data, aes(x=iteration, y=value, group=tipo, ymin=value-(2*desvio), ymax=value+(2*desvio), fill=tipo, color=tipo)) +
  geom_ribbon(alpha=.3, lty=0) +
  geom_line(size=1.5) +
  geom_vline(xintercept=326, linetype="dashed", size=.5)+
  theme(legend.justification=c(1, 1),legend.position=c(.2, .9),legend.title=element_blank())+
  theme(panel.background = element_rect(fill = 'white', colour = 'gray'),
        panel.grid.major = element_line(color = 'light gray'),
        panel.grid.minor = element_line(color = 'light gray'),
        axis.text.y = element_text(size=8),
        legend.background = element_rect(fill=alpha('white', 0.6)))+
  labs(x="Iteration", y="Accumulated Normalized Execution Time")+
  #ggtitle('b) The compilation overhead in each search iteration (Error ribbons indicate ± 2 SEM)')+
  ggtitle('       Error ribbons indicate ± 2 SEM')+
  scale_x_continuous(breaks = seq(0, 1000, 100))+
  scale_color_manual(values=c('#E07B39', '#6194C2FF'), aesthetics = c("colour", "fill"))
