abstract type AbstractMarkovModel end
abstract type AbstractDistributionModel end

"""
    MyHiddenMarkovModel <: AbstractMarkovModel

A mutable struct representing a discrete-state Hidden Markov Model (HMM).

# Fields
- `states::Array{Int64,1}`: A vector of integer identifiers for the states of the model.
- `transition::Dict{Int64, Categorical}`: A dictionary mapping each state to its transition probability distribution. The `Categorical` distribution defines the probabilities of moving to other states.
- `emission::Dict{Int64, Categorical}`: A dictionary mapping each state to its emission probability distribution. The `Categorical` distribution defines the probabilities of observing specific outputs from that state.
"""
mutable struct MyHiddenMarkovModel <: AbstractMarkovModel
    
    # --- public fields --- #
    states::Array{Int64,1}
    transition::Dict{Int64, Categorical}
    emission::Dict{Int64, Categorical}

    # --- constructor --- #
    """
        MyHiddenMarkovModel()
    
    An inner constructor to create a new, uninitialized `MyHiddenMarkovModel` instance.
    This is intended for internal use; instances should be created through a factory method.
    """
    MyHiddenMarkovModel() = new();
end

"""
    MyHiddenMarkovModelWithJumps <: AbstractMarkovModel

A mutable struct representing a Hidden Markov Model (HMM) that incorporates state jumps, extending the basic HMM with parameters for jump dynamics.

# Fields
- `states::Array{Int64,1}`: A vector of integer identifiers for the states.
- `transition::Dict{Int64, Categorical}`: A dictionary mapping each state to its transition probability distribution under normal (non-jump) conditions.
- `inverse_transition::Dict{Int64, Categorical}`: A dictionary mapping each state to its inverse transition probability, typically reversing high and low probability transitions.
- `emission::Dict{Int64, Categorical}`: A dictionary mapping each state to its emission probability distribution.
- `ϵ::Float64`: The probability of a jump occurring in any given state transition.
- `λ::Float64`: The parameter (rate) for the jump distribution, influencing the magnitude of jumps.
- `jump_distribution::Poisson`: A `Poisson` distribution that models the size or nature of the jumps, parameterized by `λ`.
"""
mutable struct MyHiddenMarkovModelWithJumps <: AbstractMarkovModel
    
    # --- public fields --- #
    states::Array{Int64,1}
    transition::Dict{Int64, Categorical}
    inverse_transition::Dict{Int64, Categorical}
    emission::Dict{Int64, Categorical}
    ϵ::Float64
    λ::Float64
    jump_distribution::Poisson

    # --- constructor --- #
    """
        MyHiddenMarkovModelWithJumps()
    
    An inner constructor to create a new, uninitialized `MyHiddenMarkovModelWithJumps` instance.
    This is intended for internal use; instances should be created through a factory method.
    """
    MyHiddenMarkovModelWithJumps() = new();
end

"""
    StudentTModel <: AbstractDistributionModel

A concrete type representing a model based on the Student's t-distribution.
Used for dispatch to select the appropriate simulation or fitting methods.
"""
struct StudentTModel <: AbstractDistributionModel end

"""
    LaplaceModel <: AbstractDistributionModel

A concrete type representing a model based on the Laplace distribution.
Used for dispatch to select the appropriate simulation or fitting methods.
"""
struct LaplaceModel <: AbstractDistributionModel end