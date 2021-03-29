using Wavelets
using LinearAlgebra
using Plots

#----------------- initializing ---------------------------
n = 1024 # original dimension of the problem
m = 100 # reduced dimension
λ = 1.5 # user defined λ parameter
x₀ = zeros(n) + 2*randn(n) # starting point
ϵ = 0.0001 # tolerance
maxiter = 10000 # hard exit condition
θₖ = [1/L] # randomly initialized θ, replace 1/L with rand() for random start.
z₀ = zeros(n) + 2*randn(n) # randomly initialized z₀
k₂ = 1 # counter
x0 = testfunction(1024, "HeaviSine") # original non-sparse signal no noise
xin = x0 + 0.3*randn(1024)  # observed signal with noise
xt = dwt(xin, wavelet(WT.sym8)) # discrete wavelet transform
xt = threshold!(xt, HardTH(), 1.0) # applying hard threshold value of 1.0
x̂ = xt # ground truth
X = [x₀] # Array to store x̂ values
Z = [z₀] # Array to store z values
Y = [] # Array to store y values


A = 10*randn(m,n) # generating A

#normalizing A such that each column has L₂ norm = 1.
for i = 1:n
    n = norm(A[:,i]) #calculate the norm of every column
    A[:,i] = (1/n)*A[:,i] #divide each column by its norm value
end

b = A*x̂ # computing b
L = λ*max(eigvals(transpose(A)*A)...) # Lipschitz constant
τ = (2/L)*rand() # step size

#-------------------- defining functions ---------------------

# closed form accelerated proximal gradient method
function accprox(θₖ, yₖ, z)
    m = (λ/2)*(Transpose(A)*(A*yₖ - b))
    n = θₖ*L
    zₖ = []
    for i in 1:length(z)
        if z[i] < (m[i] - 1)/n
            append!(zₖ, z[i]+(1-m[i])/n)
        elseif z[i] > (1+ m[i])/n
            append!(zₖ, z[i] - (1+m[i])/n)
        else
            append!(zₖ, 0)
        end
    end
    return zₖ
end


# defining f(x)
function f(x)
    return norm(x, 1) + (λ/2)*(norm(A*x - b))^2
end

# defining ∇f(x)
function ∇f(x)
    return λ*transpose(A)*(A*x - b)
end

#------------------------- main body ----------------------

@time begin
while true
    yₖ = (1 - θₖ[end])X[end] + θₖ[end]*Z[end]
    append!(Y, [yₖ])
    zₖ = accprox(θₖ[end], Y[end], Z[end])
    append!(Z, [zₖ])
    x̂ₖ = (1-θₖ[end])*X[end] + θₖ[end]*Z[end]
    append!(X, [x̂ₖ])
    θ = θₖ[end] # uncomment for constant θ case
    #θ = 0.5*(sqrt(θₖ[end]^4 + 4*θₖ[end]^2) - θₖ[end]^2) # following equation (20)
    append!(θₖ, [θ])
    global k₂ += 1
    error₂ = (norm(X[end] - X[end - 1])/norm(X[end - 1])) # computing error
    # checking exit condition
    if k₂ == maxiter
        println("reached maxiter")
        break
    elseif error₂ < ϵ
        println("convergence condition satisfied")
        break
    end
end
end

#------------------------ plotting -----------------------

#error recording for accelerated proximal gradient
fval₂ = []
err₂ = []
for i in X
    temp1 = abs(f(i)-f(x̂)) # computing error between function values
    temp2 = norm(i - x̂)/norm(x̂) # computing error between xₖ and x̂
    append!(fval₂, temp1)
    append!(err₂, temp2)
end

ans = idwt(X[end], wavelet(WT.sym8)) # applying the inverse transform

# plotting convergence error
plot(1:k₂, err₂[1:k₂], w =3, label = "APG")
xlabel!("\$ k \$")
ylabel!("error")
ylims!((0,1))

# plotting reconstructed vs original signal
plot(1:1024, xin, label="Original Signal")
plot!(1:1024, ans, label="Reconstructed signal", w =1.25)
xlabel!("t")
ylabel!("\$ x(t) \$")
