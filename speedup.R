library(ggplot2)

data <- read.csv(file = '/home/thais/Dev/TVMBench/best-seedup-2.csv', sep = ',', header = T)
# Barplot

ggplot(data, aes(x=name, y=value)) + 
  geom_violin(trim=FALSE, fill="gray")+
  theme(legend.justification=c(1, 1),legend.position=c(.9, .9),legend.title=element_blank())+
  theme(panel.background = element_rect(fill = 'white', colour = 'gray'),
        panel.grid.major = element_line(color = 'light gray'),
        panel.grid.minor = element_line(color = 'light gray'),
        axis.text.y = element_text(size=8),
        legend.background = element_rect(fill=alpha('white', 0.6)))+
  theme(axis.text.x=element_text(angle=45,hjust=.5,vjust=0.5))+
  stat_summary(fun.y=mean, geom="point", shape=23, size=2, fill="black")+
  ggtitle('a) The execution speedup in each deep learning model (both with 350 points)')+
  labs(x="Deep Learning Model", y="Speedup")+
  geom_hline(yintercept=1, size=.5, colour='black', linetype="dashed")+
  scale_y_continuous(breaks=c(seq(from=0,to=2,by=.25)))

data <- read.csv(file = '/home/thais/Dev/TVMBench/best-seedup.csv', sep = ',', header = T)
# Barplot

ggplot(data, aes(x=name, y=value)) + 
  geom_violin(trim=FALSE, fill="gray")+
  theme(legend.justification=c(1, 1),legend.position=c(.9, .9),legend.title=element_blank())+
  theme(panel.background = element_rect(fill = 'white', colour = 'gray'),
        panel.grid.major = element_line(color = 'light gray'),
        panel.grid.minor = element_line(color = 'light gray'),
        axis.text.y = element_text(size=8),
        legend.background = element_rect(fill=alpha('white', 0.6)))+
  theme(axis.text.x=element_text(angle=45,hjust=.5,vjust=0.5))+
  stat_summary(fun.y=mean, geom="point", shape=23, size=2, fill="black")+
  geom_hline(yintercept=1, size=.5, colour='black', linetype="dashed")+
  ggtitle('b) The execution speedup in each deep learning model (cache with 350 points)')+
  labs(x="Deep Learning Model", y="Speedup")



data <- read.csv(file = '/home/thais/Dev/TVMBench/acc-seedup.csv', sep = ',', header = T)

ggplot(data, aes(x=name, y=value)) + 
  coord_cartesian(ylim=c(0, 10))+
  geom_violin(trim=FALSE, fill="gray")+
  theme(legend.justification=c(1, 1),legend.position=c(.9, .9),legend.title=element_blank())+
  theme(panel.background = element_rect(fill = 'white', colour = 'gray'),
        panel.grid.major = element_line(color = 'light gray'),
        panel.grid.minor = element_line(color = 'light gray'),
        axis.text.y = element_text(size=8),
        legend.background = element_rect(fill=alpha('white', 0.6)))+
  theme(axis.text.x=element_text(angle=45,hjust=.5,vjust=0.5))+
  ggtitle('b) The compilation speedup in each deep learning model (cache with 350 points)')+
  stat_summary(fun.y=mean, geom="point", shape=23, size=2, fill="black")+
  geom_hline(yintercept=2, size=.5, colour='black', linetype="dashed")+
  geom_hline(yintercept=1, size=.5, colour='black', linetype="dashed")+
  geom_text(aes(.5,2,label = 2, vjust = -.2), colour='black') + 
  geom_text(aes(.5,1,label = 1, vjust = -.2), colour='black') + 
  labs(x="Deep Learning Model", y="Speedup")


data <- read.csv(file = '/home/thais/Dev/TVMBench/acc-seedup-2.csv', sep = ',', header = T)

ggplot(data, aes(x=name, y=value)) + 
  coord_cartesian(ylim=c(0, 10))+
  geom_violin(trim=FALSE, fill="gray")+
  theme(legend.justification=c(1, 1),legend.position=c(.9, .9),legend.title=element_blank())+
  theme(panel.background = element_rect(fill = 'white', colour = 'gray'),
        panel.grid.major = element_line(color = 'light gray'),
        panel.grid.minor = element_line(color = 'light gray'),
        axis.text.y = element_text(size=8),
        legend.background = element_rect(fill=alpha('white', 0.6)))+
  theme(axis.text.x=element_text(angle=45,hjust=.5,vjust=0.5))+
  ggtitle('a) The compilation speedup in each deep learning model (both with 350 points)')+
  stat_summary(fun.y=mean, geom="point", shape=23, size=2, fill="black")+
  geom_hline(yintercept=2, size=.5, colour='black', linetype="dashed")+
  geom_hline(yintercept=1, size=.5, colour='black', linetype="dashed")+
  geom_text(aes(.5,2,label = 2, vjust = -.2), colour='black') + 
  geom_text(aes(.5,1,label = 1, vjust = -.2), colour='black') + 
  labs(x="Deep Learning Model", y="Speedup")

