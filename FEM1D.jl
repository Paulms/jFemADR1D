include("ElementosFinitos.jl")

#Datos del Problema:
nelem = 1000
grado = 2
ϵ = 0.1
α = 1.0
β = 0
u0 = 0; u1=0
f(x) = 1

using Plots

#Resolvemos el problema
z, uh = fem1D_dar(ϵ, α, β, u0, u1, f, nelem, grado)
#Graficamos la respuesta
#Solucion exacta:
u(x,ϵ)=x - (exp(-(1-x)/ϵ)-exp(-1/ϵ))/(1-exp(-1/ϵ))
x = 0:0.01:1
plot(x, u(x,ϵ), xlab = "x", ylab = "u(x)", label = "Solución exacta",xtickfont = font(10),ytickfont = font(10))
plot!(z, uh, label = "Solución numérica", marker=:ellipse)
annotate!(0.2,u(0.45,ϵ), text("epsilon: $ϵ",10))

#Medimos el tiempo
@time fem1D_dar(ϵ, α, β, u0, u1, f, 3, 2)
@time fem1D_dar(ϵ, α, β, u0, u1, f, 500000, 2)
#@code_warntype fem1D_dar(ϵ, α, β, u0, u1, f, 3, 2)
#Profile.init(delay=0.001)
#Profile.clear()
#@profile fem1D_dar(ϵ, α, β, u0, u1, f, 500000, 2)
#using ProfileView
#ProfileView.view()
