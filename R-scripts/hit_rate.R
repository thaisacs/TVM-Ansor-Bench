library(ggplot2)

data <- read.csv(file = '/home/thais/Dev/TVMBench/csvs/hit_rate.csv', sep = ',', header = T)
# Barplot
ggplot(data, aes(x=reorder(name, -value), y=value)) + 
  theme(legend.justification=c(1, 1),legend.position=c(.9, .9),legend.title=element_blank())+
  theme(panel.background = element_rect(fill = 'white', colour = 'gray'),
        panel.grid.major = element_line(color = 'light gray'),
        panel.grid.minor = element_line(color = 'light gray'),
        axis.text.y = element_text(size=8),
        legend.background = element_rect(fill=alpha('white', 0.6)))+
  theme(axis.text.x=element_text(angle=45,hjust=.5,vjust=0.5))+
  labs(x="Deep Learning Model", y="Hit Ratio (%)")+
  geom_bar(stat = "identity", alpha=.9, color="black")+
  geom_text(aes(label = paste(signif(value, digits = 4), "%")), vjust = 1.6, color='white', size=3)

