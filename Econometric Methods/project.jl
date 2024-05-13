using DataFrames
using MarketData
using GLM
using Statistics
using CSV
using TimeSeries
using HypothesisTests
using Reduce
using Plots
function read_data(file_path)
    data = CSV.File(file_path)
    return DataFrame(data)
end
function mean_price(data)
    return mean(data.Close)
end
function compute_daily_returns(data)
    returns =  diff(log.(data.Close))
    return DataFrame(Date = data.Date[2:end], Close = returns)
end
function compute_variance(data)
    return var(data.Close)
end
function compute_volatility(data)
    return std(data.Close)
end
function compute_skewness(data)
    n = length(data)  # get the number of data points
    mean_val = mean(data)
    std_dev = std(data)
    # Ensure element-wise subtraction, raising to power, and division
    skewness = sum(((data .- mean_val) ./ std_dev).^3) / n
    return skewness
end
function compute_kurtosis(data)
    kurtosis = 1/length(data) * sum((data .- mean(data)) .^ 4) / (std(data) ^ 4)
    return kurtosis
end
function compute_jarque_bera(data)
    n = length(data)
    skewness = compute_skewness(data)
    kurtosis = compute_kurtosis(data)
    jarque_bera = (n)/6 * (skewness ^ 2 + (1/4) * (kurtosis - 3) ^ 2)
    return jarque_bera
end
#Load Information
XLV = yahoo("XLV") #Healthcare ETF
MDT = yahoo("MDT") #Medtronic
AZN = yahoo("AZN") #AstraZeneca
NVO = yahoo("NVO") #Novo Nordisk
PFE = yahoo("PFE") #Pfizer
BAYRY = yahoo("BAYRY") #Bayer
closepriceXLV = XLV[:Close]
closepriceMDT = MDT[:Close]
closepriceAZN = AZN[:Close]
closepriceNVO = NVO[:Close]
closepricePFE = PFE[:Close]
closepriceBAYRY = BAYRY[:Close]
beginning = Date(2014, 01, 01)
eind = Date(2024, 01, 01)
a = closepriceXLV[beginning:eind]
b = closepriceMDT[beginning:eind]
c = closepriceAZN[beginning:eind]
d = closepriceNVO[beginning:eind]
e = closepricePFE[beginning:eind]
f = closepriceBAYRY[beginning:eind]
# Convert and display the DataFrame
df_XLV = DataFrame(a, [:Date, :Close])
df_MDT = DataFrame(b, [:Date, :Close])
df_AZN = DataFrame(c, [:Date, :Close])
df_NVO = DataFrame(d, [:Date, :Close])
df_PFE = DataFrame(e, [:Date, :Close])
df_BAYRY = DataFrame(f, [:Date, :Close])
# Merging dataframes on the 'Date' column
portfolio_df = reduce((x, y) -> outerjoin(x, y, on = :Date, makeunique=true), [df_MDT, df_AZN, df_NVO, df_PFE, df_BAYRY])
# Add return columns for each stock
for i in 0:4  # iterating through Close and Close_1 to Close_4
    col_name = i == 0 ? :Close : Symbol("Close_$i")
    return_col_name = Symbol("Return_$i")
    portfolio_df[!, return_col_name] = [NaN; diff(log.(portfolio_df[!, col_name]))]
end
portfolio_df
# Assuming the returns columns are already in df as df.Return_0 to df.Return_4
# Create a new DataFrame to hold just the returns
returns_df = select(portfolio_df, [Symbol("Return_$i") for i in 0:4])
# Calculate the average returns for the portfolio
portfolio_returns = mean(Matrix(returns_df), dims=2) |> vec
# Assign the vector of portfolio returns to the DataFrame
portfolio_df[!, :Portfolio_Returns] = portfolio_returns
# Display to check the result
first(portfolio_df, 5)

kurtosis = compute_skewness(portfolio_df.Close_4)
plot(histogram(portfolio_df[:,:Portfolio_Returns]))

