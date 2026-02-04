# ========================================================================================= #
# Simulation Functions
# ========================================================================================= #

"""
    _simulate(m::MyHiddenMarkovModel, start::Int64, steps::Int64) -> Array{Int64,1}

Performs a standard Hidden Markov Model simulation (pure Markov process).
This method relies solely on the trained `transition` matrix and does not involve 
any jump or teleportation dynamics.

# Arguments
- `m`: The base `MyHiddenMarkovModel` struct.
- `start`: The index of the starting state.
- `steps`: The length of the simulation chain to generate.

# Returns
- `Array{Int64,1}`: A sequence of state indices representing the simulated path.
"""
function _simulate(m::MyHiddenMarkovModel, start::Int64, steps::Int64)::Array{Int64,1}

    # initialize -
    chain = Array{Int64,1}(undef, steps);
    chain[1] = start;

    # main loop -
    for i ∈ 2:steps
        # Transition to the next state based on the probability distribution 
        # of the current state (chain[i-1]).
        chain[i] = rand(m.transition[chain[i-1]]);
    end

    # return -
    return chain;
end

"""
    _simulate_poisson(m::MyHiddenMarkovModelWithJumps, start::Int64, steps::Int64) -> Array{Int64,1}

Simulates a path using the "Jump-Diffusion" logic where jumps have a specific *duration*.
Unlike standard HMMs, this method uses an exogenous `jump_distribution` (Geometric) to 
determine how long the system stays in a "jump" state before returning to normal dynamics.

# Mechanism
1. At every step, check probability `ϵ` to trigger a jump.
2. If jump triggers, sample a duration `N` from `jump_distribution`.
3. Force the system into extreme tail states (Crash or Boom) for `N` steps.
"""
function _simulate_poisson(m::MyHiddenMarkovModelWithJumps, start::Int64, steps::Int64)::Array{Int64,1}

    # initialize -
    chain = Array{Int64,1}(undef, steps);
    tmp_chain = Dict{Int64,Int64}();
    tmp_chain[1] = start;
    counter = 2;

    # main -
    jump_state = start;
    while (counter ≤ steps)
        
        # Check for Exogenous Shock (Jump)
        if (rand() < m.ϵ)

            # jump: determine the duration of the shock.
            number_of_jumps = rand(m.jump_distribution);
            
            # Define target regimes (Extreme Tails)
            number_of_states = length(m.states);
            bottom_states = [1,2,3]; # Crash / Panic states
            top_states = [number_of_states-2,number_of_states-1,number_of_states]; # Boom / Euphoria states
 
            # Force the system to stay in the tail for 'number_of_jumps'
            for _ ∈ 1:number_of_jumps
                if (rand() < 0.52)
                    tmp_chain[counter] = rand(bottom_states); # Bias slightly towards crash
                else
                    tmp_chain[counter] = rand(top_states); 
                end
                counter += 1;
                
                # safety break if we exceed simulation length
                if (counter > steps) break; end; 
            end
        else
            # Normal Operation: Follow the standard Transition Matrix
            tmp_chain[counter] = rand(m.transition[jump_state]); 
            counter += 1;
        end

        # Update the 'current' state tracker to continuity
        if (counter-1 ≤ steps)
            jump_state = tmp_chain[counter-1];
        end
    end

    # populate the final array -
    for i ∈ 1:steps
        if haskey(tmp_chain, i)
            chain[i] = tmp_chain[i];
        else
            chain[i] = chain[i-1]; # Fallback to prevent gaps
        end
    end

    # return -
    return chain;
end

"""
    _simulate_pagerank(m::MyHiddenMarkovModelWithJumps, start::Int64, steps::Int64) -> Array{Int64,1}

Simulates a path using a **Memoryless Teleportation** mechanism (inspired by Google PageRank).
Instead of a "duration" counter, this method allows for an instantaneous jump to a random 
tail state at any time step.

# Mechanism
- **Damping (1 - ϵ)**: The system follows the natural Transition Matrix.
- **Teleportation (ϵ)**: The system instantly jumps to a random state in the Crash or Boom pool.
- This creates heavy tails but lacks "stickiness" (clustering) unless the matrix itself is sticky.
"""
function _simulate_pagerank(m::MyHiddenMarkovModelWithJumps, start::Int64, steps::Int64)::Array{Int64,1}

    # initialize -
    chain = Array{Int64,1}(undef, steps);
    chain[1] = start;
    n_states = length(m.states);
    
    # Define Teleport Targets -
    crash_states = 1:3;
    boom_states = (n_states-2):n_states;

    # main loop -
    for t in 2:steps
        
        # 1. The "Google Coin Flip" (Teleportation Check)
        if (rand() < m.ϵ)
            
            # 2. Teleportation Step (Exogenous Shock)
            # Instantly move pointer to a volatile regime
            target_pool = (rand() < 0.5) ? crash_states : boom_states;
            chain[t] = rand(target_pool);
            
        else
            # 3. Damping Step (Endogenous Dynamics)
            # Follow the learned transition matrix from the previous state
            current_state = chain[t-1];
            chain[t] = rand(m.transition[current_state]);
        end
    end

    # return -
    return chain;
end

"""
    _simulate_trap(m, start, steps, trap_strength, entry_prob)

Corrected Version: Implements 'Sticky' dynamics inside the trap.
Instead of random shuffling (which kills VC), this enforces persistence
of the specific tail state, preserving the autocorrelation structure.
"""
function _simulate_trap(m::MyHiddenMarkovModelWithJumps, start::Int64, steps::Int64, 
    trap_strength::Float64, entry_prob::Float64)::Array{Int64,1}

    # initialize
    chain = Array{Int64,1}(undef, steps)
    chain[1] = start
    n_states = length(m.states)
    
    # Define Regimes
    crash_states = 1:3
    boom_states = (n_states-2):n_states
    
    # Pre-allocate for performance
    crash_indices = collect(crash_states)
    boom_indices = collect(boom_states)

    for t in 2:steps
        prev = chain[t-1]
        
        # Check Regime
        is_crash = prev in crash_states
        is_boom  = prev in boom_states
        is_tail  = is_crash || is_boom
        
        # 1. TRAP LOGIC (The Fix)
        if (is_tail && (rand() < trap_strength))
            
            # CRITICAL CHANGE: "Sticky" Logic
            # Instead of jumping to a random crash state (White Noise),
            # we have a high probability of REPEATING the exact same state.
            # This creates the "blocks" of constant volatility needed for high ACF.
            
            if rand() < 0.7 # 70% chance to freeze in the exact current state
                chain[t] = prev
            else
                # 30% chance to move, but only to a neighbor in the same regime
                # (Simulates evolving crisis)
                if is_crash
                    chain[t] = rand(crash_states)
                else
                    chain[t] = rand(boom_states)
                end
            end
            
        # 2. ENTRY LOGIC
        elseif (!is_tail && (rand() < entry_prob))
            chain[t] = rand(crash_states) # Shock entry
            
        # 3. STANDARD LOGIC
        else
            chain[t] = rand(m.transition[prev])
        end
    end

    return chain
end

# ========================================================================================= #
# Functor Interface (The Router)
# ========================================================================================= #

# Standard HMM Functor
(m::MyHiddenMarkovModel)(start::Int64, steps::Int64) = _simulate(m, start, steps);

"""
    (m::MyHiddenMarkovModelWithJumps)(start, steps; method=:standard, kwargs...)

Functor interface acting as a **Router** for different simulation methodologies.
Allows easy comparison of algorithms without changing object types.

# Methods
- `:standard` (Default): Uses `_simulate_poisson`. Best for fixed-duration shocks.
- `:pagerank`: Uses `_simulate_pagerank`. Best for memoryless, random shocks.
- `:trap`: Uses `_simulate_trap`. Best for generating Volatility Clustering (ACF decay).

# Keyword Arguments
- `trap_strength` (for `:trap`): Probability of getting stuck in a tail (default: 0.85).
- `entry_prob` (for `:trap`): Probability of slipping into a tail (default: 0.005).
"""
function (m::MyHiddenMarkovModelWithJumps)(start::Int64, steps::Int64; 
    method::Symbol = :standard, 
    trap_strength::Float64 = 0.85, 
    entry_prob::Float64 = 0.005)

    if method == :trap
        return _simulate_trap(m, start, steps, trap_strength, entry_prob);
    elseif method == :pagerank
        return _simulate_pagerank(m, start, steps);
    else
        return _simulate_poisson(m, start, steps);
    end
end


# ========================================================================================= #
# Utility / MCMC Functions
# ========================================================================================= #

"""
    learn_distribution_mcmc(model_type, returns; samples=2000)

Uses Turing.jl (Bayesian MCMC) to learn the posterior distribution of parameters 
for a given `model_type` (e.g., StudentTModel) based on empirical returns.
"""
function learn_distribution_mcmc(model_type::AbstractDistributionModel, returns::Vector{Float64}; samples::Int = 2000)
    
    # 1. Build the correct Turing model based on the input type
    model_instance = build_turing_model(model_type, returns);

    # 2. Run the NUTS (No-U-Turn Sampler) to generate the chain
    chain = Turing.sample(model_instance, NUTS(), samples);

    # return -
    return chain;
end

"""
    boost_persistence!(m, persistence_factor)

A utility for **Stress Testing**. It modifies the transition matrix in-place to 
artificially increase the 'stickiness' of tail states (Diagonal Loading).

# Arguments
- `persistence_factor`: Float (0.0 - 1.0). 
   - 0.0: No change.
   - 1.0: Tail states become absorbing (once you crash, you never leave).
"""
function boost_persistence!(m::MyHiddenMarkovModelWithJumps, persistence_factor::Float64)
    
    n_states = length(m.states);
    # Identify tail states (sorted: 1-3 Crash, (N-2)-N Boom)
    tail_indices = union(1:3, (n_states-2):n_states);

    # main loop -
    for s in tail_indices
        # Get current probabilities
        probs = m.transition[s].p;
        
        # 1. Boost the diagonal (Self-Transition)
        current_diag = probs[s];
        new_diag = current_diag + (1.0 - current_diag) * persistence_factor;
        
        # 2. Renormalize the off-diagonal elements
        off_diagonal_sum = sum(probs) - current_diag;
        
        if off_diagonal_sum > 0
            scaling_ratio = (1.0 - new_diag) / off_diagonal_sum;
            
            # Apply scaling to all, then overwrite diagonal
            new_probs = probs .* scaling_ratio;
            new_probs[s] = new_diag;
            
            # Update the model's transition matrix
            m.transition[s] = Categorical(new_probs);
        else
            # If off-diagonal was already 0, just ensure diagonal is 1.0
            new_probs = zeros(length(probs));
            new_probs[s] = 1.0;
            m.transition[s] = Categorical(new_probs);
        end
    end
    
    println("Tail state persistence boosted by factor $(persistence_factor)");
end

# ========================================================================================= #
# Growth Calculation Functions
# ========================================================================================= #

"""
    log_growth_matrix(dataset, firms; ...)

Computes the excess log returns for **multiple firms** provided in a Dictionary.
Result is a Matrix (Time x Firms).
"""
function log_growth_matrix(dataset::Dict{String, DataFrame}, 
    firms::Array{String,1}; Δt::Float64 = (1.0/252.0), risk_free_rate::Float64 = 0.0, 
    testfirm="AAPL", keycol::Symbol = :volume_weighted_average_price)::Array{Float64,2}

    # initialize -
    number_of_firms = length(firms);
    number_of_trading_days = nrow(dataset[testfirm]);
    return_matrix = Array{Float64,2}(undef, number_of_trading_days-1, number_of_firms);

    # main loop -
    for i ∈ eachindex(firms) 
        # get the firm data -
        firm_index = firms[i];
        firm_data = dataset[firm_index];

        # compute the log returns -
        for j ∈ 2:number_of_trading_days
            S₁ = firm_data[j-1, keycol];
            S₂ = firm_data[j, keycol];
            return_matrix[j-1, i] = (1/Δt)*(log(S₂/S₁)) - risk_free_rate;
        end
    end

    # return -
    return return_matrix;
end

"""
    log_growth_matrix(dataset, firm; ...)

Computes the excess log returns for a **single firm** (by ticker string) from a Dictionary.
Result is a Vector.
"""
function log_growth_matrix(dataset::Dict{String, DataFrame}, 
    firm::String; Δt::Float64 = (1.0/252.0), risk_free_rate::Float64 = 0.0, 
    keycol::Symbol = :volume_weighted_average_price)::Array{Float64,1}

    # initialize -
    number_of_trading_days = nrow(dataset[firm]);
    return_matrix = Array{Float64,1}(undef, number_of_trading_days-1);

    # get the firm data -
    firm_data = dataset[firm];

    # compute the log returns -
    for j ∈ 2:number_of_trading_days
        S₁ = firm_data[j-1, keycol];
        S₂ = firm_data[j, keycol];
        return_matrix[j-1] = (1/Δt)*log(S₂/S₁) - risk_free_rate;
    end

    # return -
    return return_matrix;
end

"""
    log_growth_matrix(dataset::DataFrame; ...)

Computes the excess log returns for a **single DataFrame**.
Useful when the data is already extracted from the dictionary.
"""
function log_growth_matrix(dataset::DataFrame; 
    Δt::Float64 = (1.0/252.0), risk_free_rate::Float64 = 0.0,
    keycol::Symbol = :volume_weighted_average_price)::Array{Float64,1}

    # initialize -
    firm_data = dropmissing(dataset, disallowmissing=true);
    number_of_trading_periods = nrow(firm_data);
    return_matrix = Array{Float64,1}(undef, number_of_trading_periods - 1);

    # compute the log returns -
    for j ∈ 2:number_of_trading_periods
        S₁ = firm_data[j-1, keycol];
        S₂ = firm_data[j, keycol];
        return_matrix[j-1] = (1/Δt)*log(S₂/S₁) - risk_free_rate;
    end

    # return -
    return return_matrix;
end

"""
    log_growth_matrix(dataset::Array{Float64,1}; ...)

Computes the excess log returns for a **raw array of prices**.
Useful for quick calculations on raw vectors.
"""
function log_growth_matrix(dataset::Array{Float64,1}; 
    Δt::Float64 = (1.0/252.0), risk_free_rate::Float64 = 0.0)::Array{Float64,1}

    # initialize -
    number_of_trading_periods = length(dataset);
    return_matrix = Array{Float64,1}(undef, number_of_trading_periods-1);

    # compute the log returns -
    for j ∈ 2:number_of_trading_periods
        S₁ = dataset[j-1];
        S₂ = dataset[j];
        return_matrix[j-1] = (1/Δt)*log(S₂/S₁) - risk_free_rate;
    end

    # return -
    return return_matrix;
end