using LinearAlgebra
using Plots
using Statistics
using GLM
using DataFrames


# Given data
y = [1649, 1435, 1864, 1892, 1913]
W = [13, 9, 17, 19, 20]

# Create a 5x2 matrix of regressors X, where we prepend a column of ones to W
one = ones(length(W))
X = hcat(one, W)

# Compute X'X
XX = X' * X

# Calculate the determinant and trace of X'X
det_XX = det(XX)
trace_XX = tr(XX)

# Compute (X'X)^(-1)
XX_inv = inv(XX)

# Estimate the OLS parameters using the formula beta_hat = (X'X)^(-1)X'y
beta_hat = XX_inv * X' * y

# Estimate the parameters by OLS without an intercept
W_matrix = reshape(W, length(W), 1)  # Convert W to a column vector
beta_hat_no_intercept = (W_matrix' * W_matrix) \ (W_matrix' * y)

# Compute the vector of residuals for both models
y_hat_with_intercept = X * beta_hat
residuals_with_intercept = y .- y_hat_with_intercept

# For the model without intercept
y_hat_no_intercept = W .* beta_hat_no_intercept
residuals_no_intercept = y .- y_hat_no_intercept

# Total Sum of Squares (TSS)
y_mean = mean(y)
TSS = sum((y .- y_mean) .^ 2)

# Plot y and W through time and save the figure
p1 = plot(1:length(y), [y, W], label=["Ice creams sold" "Temperature in Celsius"], layout=(2, 1))
savefig(p1, "time_series_plot.png")

# Scatter plot between y and W and save the figure
p2 = scatter(W, y, xlabel="Temperature in Celsius", ylabel="Ice creams sold", title="Scatter plot")
savefig(p2, "scatter_plot.png")

# Forecasting for Saturday and Sunday
W_forecast = [13, 10]  # Temperatures for Saturday and Sunday
X_forecast = hcat(ones(length(W_forecast)), W_forecast)
y_forecast = X_forecast * beta_hat
# Perform OLS regression using GLM.jl and compare with custom routine
ols_model = lm(@formula(y ~ W), DataFrame(y=y, W=W))
glm_beta_hat = coef(ols_model)

# Compute e_0'X for the model with intercept
e_0_prime_X = residuals_with_intercept' * X

# Outputs
println("GLM OLS parameters: ", glm_beta_hat)
println("e_0'X: ", e_0_prime_X)
println("OLS parameters with intercept: ", beta_hat)
println("OLS parameter without intercept: ", beta_hat_no_intercept)
println("Residuals with intercept: ", residuals_with_intercept)
println("Residuals without intercept: ", residuals_no_intercept)
println("Total Sum of Squares (TSS): ", TSS)
println("Forecast for Saturday and Sunday: ", y_forecast)