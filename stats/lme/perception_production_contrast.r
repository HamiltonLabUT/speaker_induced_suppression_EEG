library(emmeans)
library(lmerTest)

data <- read.csv('/path/to/csvs/perception_production.csv')

lm <- lmer("N100_amp ~ Cond + (1|Subject)", data=data)
em = emmeans(lm,pairwise ~ Cond, pbkrtest.limit = 11486)
summary(em)

lm <- lmer("P200_amp ~ Cond + (1|Subject)", data=data)
em = emmeans(lm,pairwise ~ Cond, pbkrtest.limit = 11486)
summary(em)

lm <- lmer("N100_latency ~ Cond + (1|Subject)", data=data)
em = emmeans(lm,pairwise ~ Cond, pbkrtest.limit = 11486)
summary(em)

lm <- lmer("P200_latency ~ Cond + (1|Subject)", data=data)
em = emmeans(lm,pairwise ~ Cond, pbkrtest.limit = 11486)
summary(em)

lm <- lmer("peak_to_peak ~ Cond + (1|Subject)", data=data)
em = emmeans(lm,pairwise ~ Cond, pbkrtest.limit = 11486)
summary(em)