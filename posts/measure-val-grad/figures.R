library(tidyverse)
library(latex2exp)
library(ggpubr)

###Functions

mu <- 1
sigma <- 1
eps <- 0.1
x <- seq(-3, 5, by = 0.01)

f_norm <- function(x, mu, sigma) {
  1 / (sigma * sqrt(2 * pi)) * exp(-(x - mu) ^ 2 / (2 * sigma ^ 2))
}

f_norm_dmu <- function(x, mu, sigma) {
  (x - mu) / (sigma ^ 3 * sqrt(2 * pi)) * exp(-(x - mu) ^ 2 / (2 * sigma ^ 2))
}

f_norm_mu_pplus <- function(x, mu, sigma) {
  (x - mu) / (sigma ^ 3 * sqrt(2 * pi)) * exp(-(x - mu) ^ 2 / (2 * sigma ^ 2)) * (x >= mu) 
}

f_norm_mu_pminus <- function(x, mu, sigma) {
  -(x - mu) / (sigma ^ 3 * sqrt(2 * pi)) * exp(-(x - mu) ^ 2 / (2 * sigma ^ 2)) * (x < mu) 
}

f_norm_sigma_pplus <- function(x, mu, sigma) {
  (x - mu)^2 / (sigma ^ 3 * sqrt(2 * pi)) * exp(-(x - mu) ^ 2 / (2 * sigma ^ 2)) / sigma
}

f_norm_sigma_pminus <- function(x, mu, sigma) {
  1 / (sigma * sqrt(2 * pi)) * exp(-(x - mu) ^ 2 / (2 * sigma ^ 2)) / sigma
}

f_norm_dsigma <- function(x, mu, sigma) {
  f_norm_sigma_pplus(x, mu, sigma) - f_norm_sigma_pminus(x, mu, sigma)
}

###Plot

### mu
df_plot <- data.frame(
  x = x,
  y_norm = f_norm(x, mu, sigma),
  y_eps_norm = f_norm(x, mu + eps, sigma),
  y_norm_dmu = f_norm_dmu(x, mu, sigma),
  y_norm_pplus = f_norm_mu_pplus(x, mu, sigma),
  y_norm_pminus = f_norm_mu_pminus(x, mu, sigma)
)

p_norm_mu <-
  ggplot() + 
  geom_line(aes(x = x, y = y_norm), alpha = 0.6, data = df_plot) + 
  geom_line(
    aes(x = x, y = y_eps_norm),
    linetype = 2,
    alpha = 0.6,
    data = df_plot
  ) + 
  coord_cartesian(xlim=c(-2, 4)) + 
  xlab("") + 
  ylab(TeX("$p_N (x; \\mu, \\sigma)$"))

p_norm_dmu <-
  ggplot(aes(x = x, y = y_norm_dmu), data = df_plot) + 
  geom_hline(yintercept=0, linetype=4) + 
  geom_line(alpha = 0.6) + 
  coord_cartesian(xlim=c(-2, 4)) + 
  xlab("") + 
  ylab(TeX("$\\nabla_{\\mu} p_N (x; \\mu, \\sigma)$"))

p_norm_mu_pplus <-
  ggplot(aes(x = x, y = y_norm_pplus), data = df_plot) + 
  geom_line(alpha = 0.6) + 
  coord_cartesian(xlim=c(-2, 4)) +   
  xlab("") + 
  ylab(TeX("$c_{\\mu} p_N^+ (x; \\mu, \\sigma)$"))

p_norm_mu_pminus <-
  ggplot(aes(x = x, y = y_norm_pminus), data = df_plot) + 
  geom_line(alpha = 0.6) + 
  coord_cartesian(xlim=c(-2, 4)) +   
  xlab(TeX("$x$")) + 
  ylab(TeX("$c_{\\mu} p_N^- (x; \\mu, \\sigma)$"))

ggarrange(p_norm_mu, p_norm_dmu, p_norm_mu_pplus, p_norm_mu_pminus, ncol = 1, align = "v")

ggsave("norm_mu.png", width = 5, height = 7)

### sigma
df_plot <- data.frame(
  x = x,
  y_norm = f_norm(x, mu, sigma),
  y_eps_norm = f_norm(x, mu, sigma + eps),
  y_norm_dsigma = f_norm_dsigma(x, mu, sigma),
  y_norm_pplus = f_norm_sigma_pplus(x, mu, sigma),
  y_norm_pminus = f_norm_sigma_pminus(x, mu, sigma)
)

p_norm_sigma <-
  ggplot() + 
  geom_line(aes(x = x, y = y_norm), alpha = 0.6, data = df_plot) + 
  geom_line(
    aes(x = x, y = y_eps_norm),
    linetype = 2,
    alpha = 0.6,
    data = df_plot
  ) + 
  coord_cartesian(xlim=c(-2, 4)) + 
  xlab("") + 
  ylab(TeX("$p_N (x; \\mu, \\sigma)$"))

p_norm_dsigma <-
  ggplot(aes(x = x, y = y_norm_dsigma), data = df_plot) + 
  geom_hline(yintercept=0, linetype=4) + 
  geom_line(alpha = 0.6) + 
  coord_cartesian(xlim=c(-2, 4)) + 
  xlab("") + 
  ylab(TeX("$\\nabla_{\\sigma} p_N (x; \\mu, \\sigma)$"))

p_norm_sigma_pplus <-
  ggplot(aes(x = x, y = y_norm_pplus), data = df_plot) + 
  geom_line(alpha = 0.6) + 
  coord_cartesian(xlim=c(-2, 4)) +   
  xlab("") + 
  ylab(TeX("$c_{\\sigma} p_N^+ (x; \\mu, \\sigma)$"))

p_norm_sigma_pminus <-
  ggplot(aes(x = x, y = y_norm_pminus), data = df_plot) + 
  geom_line(alpha = 0.6) +
  coord_cartesian(xlim=c(-2, 4)) +    
  xlab(TeX("$x$")) + 
  ylab(TeX("$c_{\\sigma} p_N^- (x; \\mu, \\sigma)$"))

ggarrange(p_norm_sigma, p_norm_dsigma, p_norm_sigma_pplus, p_norm_sigma_pminus, ncol = 1, align = "v")

ggsave("norm_sigma.png", width = 5, height = 7)
