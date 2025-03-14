# Rust (1987) bus engine replacement problem
# Notation: I use underscore to denote empirical counterparts

# Import packages
using Optim
using Distributions
using Statistics
using DataFrames
using CSV
using LinearAlgebra
using Plots

function compute_U(θ::Vector, s::Vector)::Matrix
    """Compute static utility"""
    u1 = - θ[1]*s - θ[2]*s.^2       # Utility of not investing
    u2 = - θ[3]*ones(size(s))       # Utility of investing
    U = [u1 u2]                     # Combine in a matrix
    return U
end;

function compute_Vbar(θ::Vector, λ::Number, β::Number, s::Vector)::Matrix
    """Compute value function by Bellman iteration"""
    k = length(s)                                 # Dimension of the state space
    U = compute_U(θ, s)                           # Static utility
    index_λ = Int[1:k [2:k; k]];                  # Mileage index
    index_A = Int[1:k ones(k,1)];                 # Investment index
    γ = Base.MathConstants.eulergamma             # Euler's gamma

    # Iterate the Bellman equation until convergence
    Vbar = zeros(k, 2);
    Vbar1 = Vbar;
    dist = 1;
    iter = 0;
    while dist>1e-8
        V = γ .+ log.(sum(exp.(Vbar), dims=2))     # Compute value
        expV = V[index_λ] * [1-λ; λ]               # Compute expected value
        Vbar1 =  U + β * expV[index_A]             # Compute v-specific
        dist = max(abs.(Vbar1 - Vbar)...);         # Check distance
        iter += 1;
        Vbar = Vbar1                               # Update value function
    end
    return Vbar
end;

function generate_data(θ::Vector, λ::Number, β::Number, s::Vector, N::Int)::Tuple
    """Generate data from primitAes"""
    Vbar = compute_Vbar(θ, λ, β, s)             # Solve model
    ε = rand(Gumbel(0,1), N, 2)                 # Draw shocks
    St = rand(s, N)                             # Draw states
    A = (((Vbar[St,:] + ε) * [-1;1]) .> 0)     # Compute investment decisions
    δ = (rand(Uniform(0,1), N) .< λ)            # Compute mileage shock
    St1 = min.(St .* (A.==0) + δ, max(s...))   # Compute nest state
    df = DataFrame(St=St, A=A, St1=St1)
    CSV.write("data/rust.csv", df)
    return St, A, St1, df
end;

function logL_Rust(θ0::Vector, λ::Number, β::Number, s::Vector, St::Vector, A::BitVector)::Number
    """Compute log-likelihood functionfor Rust problem"""
    # Compute value
    Vbar = compute_Vbar(θ0, λ_, β, s)

    # Implied choice probabilities
    EP = exp.(Vbar[:,2]) ./ (exp.(Vbar[:,1]) + exp.(Vbar[:,2]))

    # Likelihood
    logL = sum(log.(EP[St[A.==1]])) + sum(log.(1 .- EP[St[A.==0]]))
    return -logL
end;

function compute_T(k::Int, λ_::Number)::Array
    """Compute transition matrix"""
    T = zeros(k, k, 2);

    # Conditional on not investing
    T[k,k,1] = 1;
    for i=1:k-1
        T[i,i,1] = 1-λ_
        T[i,i+1,1] = λ_
    end

    # Conditional on investing
    T[:,1,2] .= 1-λ_;
    T[:,2,2] .= λ_;

    return(T)
end;

## Main

# Set parameters
θ = [0.13; -0.004; 3.1];
λ = 0.82;
β = 0.95;

# State space
k = 50;
s = Vector(1:10);

# Compute value function
Vbar = compute_Vbar(θ, λ, β, s);

# Generate data
N = Int(1e5);
St, A, St1, df = generate_data(θ, λ, β, s, N);
print("\n\nwe observe ", sum(A), " investment decisions in ", N, " observations")

# Estimate lambda
Δ = St1 - St;
λ_ = mean(Δ[(A.==0) .& (St.<10)]);

print("\n\nEstimated lambda: $λ_ (true = $λ)")

# True likelihood value
logL_trueθ = logL_Rust(θ, λ, β, s, St, A);
print("\n\nThe likelihood at the true parameter is $logL_trueθ")

# Select starting values
θ0 = Float64[0,0,0];

opt_options = Optim.Options(show_trace = true, store_trace = true,extended_trace = true)

# Optimize
result = optimize(x -> logL_Rust(x, λ, β, s, St, A), θ0, opt_options);

θ_R = result.minimizer;

result.trace

df = DataFrame(result.trace)

#Extract the iterated theta values
centroid_vector = df.metadata .|> x -> x["centroid"]
centroid_matrix = hcat(centroid_vector...)'

# Extract columns
theta_1 = centroid_matrix[:, 1]
theta_2 = centroid_matrix[:, 2]
theta_3 = centroid_matrix[:, 3]

# Get iteration index
iterations = 1:length(theta_1)

# Plot theta values
plot(iterations, theta_1, label="θ₁", linewidth=2)
plot!(iterations, theta_2, label="θ₂", linewidth=2)
plot!(iterations, theta_3, label="θ₃", linewidth=2)

# Add horizontal dashed lines at specified values (without legend)
hline!([0.13], linestyle=:dash, color=:black, label=nothing)
hline!([-0.004], linestyle=:dash, color=:black, label=nothing)
hline!([3.1], linestyle=:dash, color=:black, label=nothing)

# Labels and title
xlabel!("Iteration")
ylabel!("Theta Values")
title!("Theta Evolution Over Iterations")

# Save the plot
savefig("theta_evolution.png")

print("\n\nEstimated thetas: $θ_R (true = $θ)")


