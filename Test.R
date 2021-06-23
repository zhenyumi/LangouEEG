library(ggplot2)
setwd("C:\\MatlabWorkspace\\Light")
      
flicker <- read.csv("ratios_flicker.csv", header = TRUE)
rest <- read.csv("ratios_rest.csv", header = TRUE)
flicker_all <- read.csv("ratios_flicker_all.csv", header = TRUE)
rest_all <- read.csv("ratios_rest_all.csv", header = TRUE)
rest_MA <- read.csv("ratiosMA_rest.csv", header = TRUE)
rest_MA_all <- read.csv("ratiosMA_rest_all.csv", header = TRUE)
flicker_MA <- read.csv("ratiosMA_flicker.csv", header = TRUE)
flicker_MA_all <- read.csv("ratiosMA_flicker_all.csv", header = TRUE)

ggplot(flicker_all) +
  aes(x = "", y = ratios, fill = labels) +
  geom_boxplot(shape = "circle") +
  scale_fill_manual(values = list(
    ratio_DU = "#F8766D", ratio_TD = "#00C19F", ratio_TU = "#FF61C3")) +
  labs(title = "flicker") +
  theme_minimal() +
  facet_wrap(vars(labels))

ggplot(rest_all) +
  aes(x = "", y = ratios, fill = labels) +
  geom_boxplot(shape = "circle") +
  scale_fill_manual(values = list(
    ratio_DU = "#F8766D", ratio_TD = "#00C19F", ratio_TU = "#FF61C3")) +
  labs(title = "rest") +
  theme_minimal() +
  facet_wrap(vars(labels))

attach(flicker)
t.test(ratio_TD, ratio_DU)
t.test(ratio_TU, ratio_DU)
t.test(ratio_TD, ratio_TU)
detach(flicker)

attach(rest)
t.test(ratio_TD, ratio_DU)
t.test(ratio_TU, ratio_DU)
t.test(ratio_TD, ratio_TU)
detach(rest)

t.test(flicker$ratio_TD, rest$ratio_TD)
t.test(flicker$ratio_TU, rest$ratio_TU)
t.test(flicker$ratio_DU, rest$ratio_DU)

t.test(flicker_MA$ratio_TD, rest_MA$ratio_TD)
t.test(flicker_MA$ratio_TU, rest_MA$ratio_TU)
t.test(flicker_MA$ratio_DU, rest_MA$ratio_DU)

attach(flicker_MA)
t.test(ratio_TD, ratio_DU)
t.test(ratio_TU, ratio_DU)
t.test(ratio_TD, ratio_TU)
detach(flicker_MA)

attach(rest_MA)
t.test(ratio_TD, ratio_DU)
t.test(ratio_TU, ratio_DU)
t.test(ratio_TD, ratio_TU)
detach(rest_MA)
