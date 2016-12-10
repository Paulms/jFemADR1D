#ElementosFinitos.jl
# Finite Elements method for the advection-diffusion-reaction problem
# -ϵu''(x)+αu'(x)+βu(x)=f(x)   x∈[a,b]
# u(a) = u0, u(b) = u1
# Solution of the 1D problem
# Elements available: picewise polynomials of first and second order
using ForwardDiff
using FastGaussQuadrature

function get_basis(nod::Int, dims::Int)
    if nod == 2
        Base = :(x::Real->[(1-x)/2, (1+x)/2]);
    elseif nod == 3
        Base = :(x::Real->[-x*(1-x)/2, (1+x)*(1-x), x*(1+x)/2]);
    else
         error("Polinomial not available")
    end
    return Base, :(y::Real->ForwardDiff.derivative($Base, y));
end

function get_Kt_local(ϵ::Real, α::Real, β::Real, coords::Vector, P, dP, nodes::Vector, weights::Vector)
    nod = length(coords)
    kt = zeros(nod,nod)
    PP = zeros(1,nod)
    detB = (coords[end]-coords[1])/2
    Bt = 1/detB
    for i in 1:length(nodes)
        DeP = Bt*dP(nodes[i])'
        PP[1,:] = P(nodes[i])
        difusion = ϵ*DeP'*DeP;
        adveccion = α*PP'*DeP;
        reaccion = β*PP'*PP;
        kt = kt + (difusion + adveccion + reaccion)*weights[i];
    end
    return kt*detB
end

function get_Ft_local(coords::Vector, P, dP, nodes::Vector, weights::Vector)
    nod = length(coords)
    kt = zeros(nod,nod)
    PP = zeros(1,nod)
    detB = (coords[end]-coords[1])/2
    Bt = 1/detB
    for i in 1:length(nodes)
        PP[1,:] = P(nodes[i])
        reaccion = PP'*PP;
        kt = kt + reaccion*weights[i];
    end
    return kt*detB
end

function fem1D_dar(ϵ::Real, α::Real, β::Real, u0::Real, u1::Real, f, nelem::Int, grado::Int, npoints::Int=6)
    nnodos = grado*nelem + 1
    h = 1/(nelem*grado)
    coordenadas = collect(0:h:1)
    conectividad = [j+i for i in 1:grado+1, j in 0:grado:grado*nelem-1]
    refnodo = zeros(nnodos);refnodo[1] = 1; refnodo[nnodos]=1;
    refele = ones(nelem);
    #Obtener Base según método a usar
    nod = grado + 1
    P, dP = get_basis(nod, 1)
    P = eval(P); dP = eval(dP)

    #Obtener datos para integracion
    nodes, weights = gausslegendre(npoints);

    #Ensamblar Sistema
    iA = Array{Int64}(nelem,nod,nod); jA = Array{Int64}(nelem,nod,nod); sA = zeros(nelem,nod,nod);
    #A = spzeros(nnodos,nnodos);
    F = zeros(nnodos);
    Ft_local = zeros(grado+1);
    Kt_local = zeros(grado+1,grado+1)
    i1 = 0
    j1 = 0
    #if partition is uniform we only need to calculate Kt_local once
    a1 = 1
    a2 = grado + 1
    Ft_local = get_Ft_local(coordenadas[a1:a2],P,dP,nodes, weights)*
    [f(coordenadas[i]) for i in a1:a2]
    Kt_local = get_Kt_local(ϵ, α, β, coordenadas[a1:a2],P,dP,nodes, weights)
    for elem = 1:nelem
        for i = 1:(grado+1)
            i1 = conectividad[i,elem]
            for j = 1:(grado+1)
                j1 = conectividad[j,elem]
                #A[i1, j1] = A[i1, j1] + Kt_local[i,j]
                iA[elem,i,j] = i1;
                jA[elem,i,j] = j1;
                sA[elem,i,j] = Kt_local[i,j]
            end
            F[i1] = F[i1] + Ft_local[i]
        end
    end
    A = sparse(vec(iA), vec(jA), vec(sA))
    #Imponer condiciones de Dirichlet
    A[1,:] = 0;    A[1,1] = 1
    A[nnodos,:] = 0;    A[nnodos,nnodos] = 1
    F[1] = u0;    F[nnodos] = u1
    # Resolver el problema y entregar respuesta
    uh = A\F;
    return coordenadas, uh
end
