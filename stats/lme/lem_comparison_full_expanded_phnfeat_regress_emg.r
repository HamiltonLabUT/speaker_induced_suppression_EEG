library(emmeans)
library(lmerTest)

data <- read.csv('/path/to/csvs/model1_model2.csv')

lm <- lmer("r ~ model + (1|subject)", data=data)

em = emmeans(lm,pairwise ~ model)
summary(em)
