using Random
using Distributions
using DataFrames
using GLM
using LinearAlgebra
using Statistics
Random.seed!(666)
n = 100
educ = rand(Uniform(0,15), n)
eps = 100 * randn(n)
scatter(1:n, eps)
wage = 500 .+ 100 * educ .+ eps

println("Wages:")
for i in 1:5
    println(wage[i])
end

eps_low = 2*randn(50)
eps_high = 5*randn(50)

eps2 = vcat(eps_low, eps_high)
scatter(1:100, eps2)
savefig("scatter_plot.png")

wage = 500 .+ 100 * educ .+ eps2

println("Wages:")
for i in 1:5
    println(wage[i])
end
n = 100
educ = rand(Uniform(0, 15), n)
male = rand(Binomial(1, 0.5), n)  # Randomly assign gender
female = 1 .- male  # Female is the complement of Male

X = hcat(ones(n), educ, male, female)  # Including an intercept, education, and gender dummies

println("First 10 rows of matrix X:")
println(X[1:10, :])
