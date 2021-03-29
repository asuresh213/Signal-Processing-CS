using LinearAlgebra
using Plots

#----------------- initializing ---------------------------
n = 1000 # original dimension of the problem
m = 100 # reduced dimension
p = 10 # sparsity factor of x̂
λ = 1.5 # user defined λ parameter
x₀ = zeros(n) + 2*randn(n) # starting point
ϵ = 0.0001 # tolerance
maxiter = 10000 # hard exit condition
k = 1 # counter
x = [x₀] # array of xₖ

#generating x̂
x̂ = zeros(n)
for i = 1:p
    index = rand(1:n)
    x̂[index] = rand(-100:100) # uniformly distributed random number
end

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

# defining prox_τϕ
function prox(z)
    temp = z >= τ ? z-τ : z <= -τ ? z + τ : 0
    return temp
end

# defining f(x)
function f(x)
    return norm(x, 1) + (λ/2)*(norm(A*x - b))^2
end

# defining ∇f(x)
function ∇f(x)
    return λ*transpose(A)*(A*x - b)
end

#------------------------ main body ---------------------
@time begin
# main loop for proximal gradient descent
while true
    xₖ = prox.(x[end] - τ*∇f(x[end])) # implementing (5)
    append!(x, [xₖ]) # adding new point to our array
    global k += 1 # increasing counter
    error = (norm(x[end] - x[end - 1])/norm(x[end - 1])) # computing error
    # checking exit condition
    if k == maxiter
        println("reached maxiter! ")
        break
    elseif error < ϵ
        println("convergence condition satisfied! ")
        break
    end
end
end


#--------------- accelerated prox gradient ----------------

#------------------- initiailizing ------------------------

θₖ = [1/L] # randomly initialized θ, replace 1/L with rand() for random start.
z₀ = zeros(n) + 2*randn(n) # randomly initialized z₀
X = [x₀]
Z = [z₀]
Y = []
k₂ = 1

#----------------- defining functions ----------------------

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


# Error recording for regular proximal gradient
fval = []
err = []
for i in x
    temp1 = abs(f(i) - f(x̂)) # computing error between function values
    temp2 = norm(i - x̂)/norm(x̂) # computing error between xₖ and x̂
    append!(fval, temp1)
    append!(err, temp2)
end

#error recording for accelerated proximal gradient
fval₂ = []
err₂ = []
for i in X
    temp1 = abs(f(i)-f(x̂)) # computing error between function values
    temp2 = norm(i - x̂)/norm(x̂) # computing error between xₖ and x̂
    append!(fval₂, temp1)
    append!(err₂, temp2)
end


min_iter = min(k, k₂) # to avoid indexing errors

#plotting O(1/k)
t₁ = []
for i in 1:k
    append!(t₁, 100000/(i-1+0.01))
end

t₂ = []
#plotting O(1/√k)
for i in 1:k₂
    append!(t₂, 1000/sqrt(i-1+0.01))
end


# plotting APG curve against c/k
Plots.GRBackend()
plot(1:k, fval, w = 3, label = "RPG")
plot!(1:k, t₁, w = 2, linestyle=:dash, label = "\$ c/\\sqrt{k}\$")
xlabel!("\$ k \$")
ylabel!("\$ |f(x_k) - f(x_*)|\$")
ylims!((0,1130))


# plotting APG curve against c/√k
Plots.GRBackend()
plot(1:k₂, fval₂, w = 3, label = "APG")
plot!(1:k₂, t₂, w = 2, linestyle=:dash, label = "\$ c/\\sqrt{k}\$")
xlabel!("\$ k \$")
ylabel!("\$ |f(x_k) - f(x_*)|\$")
ylims!((0,1130))


# plotting difference between estimated and actual function values
Plots.GRBackend()
plot(1:min_iter, fval₂[1:min_iter], w=3, label = "APG")
plot!(1:min_iter, fval[1:min_iter], w = 3, label = "RPG")
xlabel!("\$ k \$")
ylabel!("\$ |f(x_k) - f(x_*)|\$")
ylims!((0,1100))

# plotting convergence error
plot(1:min_iter, err₂[1:min_iter], w =3, label = "APG")
plot!(1:min_iter, err[1:min_iter], w= 3, legend=:bottomleft, label = "RPG")
xlabel!("\$ k \$")
ylabel!("error")
ylims!((0,1))
