library(ggplot2)

data <- read.csv(file = '/home/thais/Dev/TVMBench/test-4.csv', sep = ',', header = T)
# Barplot


ggplot(data, aes(x = reorder(name, -value), y = value)) + 
  geom_bar(stat = "identity")+
  theme(legend.justification=c(1, 1),legend.position=c(.9, .9),legend.title=element_blank())+
  theme(panel.background = element_rect(fill = 'white', colour = 'gray'),
        panel.grid.major = element_line(color = 'light gray'),
        panel.grid.minor = element_line(color = 'light gray'),
        axis.text.y = element_text(size=8),
        legend.background = element_rect(fill=alpha('white', 0.6)))+
  theme(axis.text.x=element_text(angle=45,hjust=.5,vjust=0.5))+
  geom_hline(yintercept=1, size=.5, colour='black', linetype="dashed")+
  geom_text(aes(.5,1,label = 1, vjust = -.2), colour='black')+
  geom_errorbar(aes(ymin=value-sd, ymax=value+sd), position = position_dodge(0.9), width = 0.25)+
  labs(x="Deep Learning Model", y="Speedup")+
  geom_text(aes(label = paste(signif(value, digits = 3))), vjust = -.6, hjust=-.3, color='black', size = 4)
