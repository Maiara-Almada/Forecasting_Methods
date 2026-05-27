# Run Rmd code chunks to get exact output values including the new model

library(tidyverse)
library(tsibble)
library(fable)
library(feasts)
library(urca)
library(lubridate)

# Load data
unemployment_ts <- read.csv("UNRATE.csv") %>%
  mutate(observation_date = yearmonth(observation_date)) %>%
  as_tsibble(index = observation_date)

# Split into Training Data (All data prior to 2019)
train_data <- unemployment_ts %>% 
  filter(year(observation_date) < 2019)

# Split into Test Data (Exactly the year 2019)
test_data <- unemployment_ts %>% 
  filter(year(observation_date) > 2018 & year(observation_date) < 2020)

cat("\n--- 3.1 Candidate Model Specification ---\n")
fit <- train_data %>%
  model(
    sarima111113 = ARIMA(UNRATE ~ pdq(1, 1, 1) + PDQ(1, 1, 3)),  
    sarima110113 = ARIMA(UNRATE ~ pdq(1, 1, 0) + PDQ(1, 1, 3)),  
    sarima111111 = ARIMA(UNRATE ~ pdq(1, 1, 1) + PDQ(1, 1, 1)),
    sarima210113 = ARIMA(UNRATE ~ pdq(2, 1, 0) + PDQ(1, 1, 3)),
    sarima211112 = ARIMA(UNRATE ~ pdq(2, 1, 1) + PDQ(1, 1, 2))
  )

print(glance(fit) %>% select(.model, sigma2, log_lik, AIC, AICc, BIC))

cat("\n--- Coefficients for sarima211112 ---\n")
print(fit %>% select(sarima211112) %>% report())

cat("\n--- 4.1 Formal Diagnostic Tests (Ljung-Box) ---\n")
# ljung_box test
lb_results <- bind_rows(
  augment(fit) %>% filter(.model == "sarima111113") %>% features(.innov, ljung_box, lag = 36, dof = 6) %>% mutate(.model = "sarima111113"),
  augment(fit) %>% filter(.model == "sarima110113") %>% features(.innov, ljung_box, lag = 36, dof = 5) %>% mutate(.model = "sarima110113"),
  augment(fit) %>% filter(.model == "sarima111111") %>% features(.innov, ljung_box, lag = 36, dof = 4) %>% mutate(.model = "sarima111111"),
  augment(fit) %>% filter(.model == "sarima210113") %>% features(.innov, ljung_box, lag = 36, dof = 6) %>% mutate(.model = "sarima210113"),
  augment(fit) %>% filter(.model == "sarima211112") %>% features(.innov, ljung_box, lag = 36, dof = 6) %>% mutate(.model = "sarima211112")
) %>% arrange(desc(lb_pvalue))
print(lb_results)

cat("\n--- 5. In-Sample Forecast and Fit Errors ---\n")
in_sample_acc <- accuracy(fit) %>%
  select(.model, RMSE, MAE, MAPE, MASE) %>%
  arrange(RMSE)
print(in_sample_acc)

cat("\n--- 6. Out-of-Sample Forecast and Validation ---\n")
fc_all <- fit %>% forecast(new_data = test_data)
out_sample_acc <- accuracy(fc_all, unemployment_ts) %>%
  select(.model, RMSE, MAE, MAPE, MASE) %>%
  arrange(RMSE)
print(out_sample_acc)

cat("\n--- 7. Forecast Combination ---\n")
valid_models <- augment(fit) %>%
  features(.innov, ljung_box, lag = 24) %>%
  filter(lb_pvalue > 0.00001) %>%
  pull(.model)

weights_rmse <- accuracy(fit) %>%
  filter(.model %in% valid_models) %>%
  mutate(w = (1/RMSE) / sum(1/RMSE)) %>%
  select(.model, RMSE, w)
print(weights_rmse)

fc_combo_rmse <- fc_all %>%
  filter(.model %in% valid_models) %>%
  as_tibble() %>%
  left_join(weights_rmse %>% select(.model, w), by = ".model") %>%
  group_by(observation_date) %>%
  summarise(.mean = sum(w * .mean), .groups = "drop") %>%
  mutate(.model = "combo_rmse")

combo_rmse_val <- fc_combo_rmse %>%
  inner_join(test_data %>% as_tibble(), by = "observation_date") %>%
  summarise(.model = "combo_rmse", RMSE = sqrt(mean((UNRATE - .mean)^2)))
print(combo_rmse_val)
