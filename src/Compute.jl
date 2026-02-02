function _simulate(m::MyHiddenMarkovModel, start::Int64, steps::Int64)::Array{Int64,1}

    # initialize -
    chain = Array{Int64,1}(undef, steps);
    chain[1] = start;

    # main loop -
    for i ∈ 2:steps
        chain[i] = rand(m.transition[chain[i-1]]);
    end

    return chain;
end

function _simulate(m::MyHiddenMarkovModelWithJumps, start::Int64, steps::Int64)::Array{Int64,1}

    # initialize -
    chain = Array{Int64,1}(undef, steps);
    tmp_chain = Dict{Int64,Int64}();
    tmp_chain[1] = start;
    counter = 2;

    # main -
    jump_state = start;
    while (counter ≤ steps)
        
        if (rand() < m.ϵ)

            # # jump: find the next state. It is lowest probability state from here
            number_of_jumps = rand(m.jump_distribution);
            number_of_states = length(m.states);
            bottom_states = [1,2,3]; # super bad
            top_states = [number_of_states-2,number_of_states-1,number_of_states]; # super good
 
            @show number_of_jumps

            for _ ∈ 1:number_of_jumps
                if (rand() < 0.52)
                    tmp_chain[counter] = rand(bottom_states) # a jump transition to bottom states
                else
                    tmp_chain[counter] = rand(top_states) # a jump transition to top states
                end
                counter += 1;
            end
        else
            tmp_chain[counter] = rand(m.transition[jump_state]); # a normal transition
            counter += 1; # increment counter
        end

        jump_state = tmp_chain[counter-1]; # get the last state
    end

    # populate the chain from tmp_chain -
    for i ∈ 1:steps
        chain[i] = tmp_chain[i];
    end

    # return -
    return chain;
end


"""
    _simulate_pagerank(m::MyHiddenMarkovModelWithJumps, start::Int64, steps::Int64) -> Array{Int64,1}

Private method: Simulates a path using Google's PageRank 'Teleportation' logic.
Replaces the Poisson duration with an instantaneous probability check at every step.
"""
function _simulate_pagerank(m::MyHiddenMarkovModelWithJumps, start::Int64, steps::Int64)::Array{Int64,1}

    # initialize
    chain = Array{Int64,1}(undef, steps)
    chain[1] = start
    
    n_states = length(m.states)
    
    # Define Teleport Targets (The "Personalized" Vector)
    # We maintain your logic of targeting extreme tails
    crash_states = 1:3
    boom_states = (n_states-2):n_states

    # Main Loop (Standard 1 to steps, no complex counters)
    for t in 2:steps
        
        # 1. The "Google Coin Flip"
        # random() < epsilon means "Teleport" (1 - Damping Factor)
        if (rand() < m.ϵ)
            
            # 2. Teleportation Step
            # We instantly jump to a specific regime (Crash or Boom).
            # Unlike the previous version, there is no 'duration' loop.
            # We are just moving the pointer to a volatile state.
            
            target_pool = (rand() < 0.5) ? crash_states : boom_states
            chain[t] = rand(target_pool)
            
        else
            # 3. Damping Step (Normal Transition)
            # Follow the learned transition matrix from the previous state
            current_state = chain[t-1]
            chain[t] = rand(m.transition[current_state])
        end
    end

    return chain
end


"""
    _simulate_trap(m::MyHiddenMarkovModelWithJumps, start::Int64, steps::Int64) -> Array{Int64,1}

Simulates a path where 'Teleportation' acts as a gravity well (Trap) inside tail states.
This enforces volatility clustering (persistence) without affecting normal market dynamics.
"""
function _simulate_trap(m::MyHiddenMarkovModelWithJumps, start::Int64, steps::Int64)::Array{Int64,1}
    chain = Array{Int64,1}(undef, steps)
    chain[1] = start
    
    n_states = length(m.states)
    
    # Define The "Trap" Zones
    crash_states = 1:3
    boom_states = (n_states-2):n_states
    
    # The "Trap Strength" (Probability of getting stuck in the regime)
    # You can reuse m.ϵ or define a new parameter. 
    # Let's assume we want a HIGH persistence, e.g., 0.8
    trap_strength = 0.8 

    for t in 2:steps
        prev_state = chain[t-1]
        
        # 1. Check if we are currently in a "Trap Zone"
        in_crash = prev_state in crash_states
        in_boom  = prev_state in boom_states
        
        # 2. Apply Trap Logic (Conditional Teleportation)
        if (in_crash || in_boom) && (rand() < trap_strength)
            
            # We are TRAPPED. Teleport to a random state within the SAME regime.
            # This creates the "Clustering" effect.
            if in_crash
                chain[t] = rand(crash_states)
            else
                chain[t] = rand(boom_states)
            end
            
        else
            # 3. Normal Dynamics (or Failed Trap)
            # Follow the learned transition matrix.
            # This allows for:
            #   a) Normal evolution in normal times.
            #   b) Natural "Entry" into crashes from normal states.
            #   c) Natural "Exit" from crashes (if the trap coin flip failed).
            chain[t] = rand(m.transition[prev_state])
        end
    end

    return chain
end

(m::MyHiddenMarkovModel)(start::Int64, steps::Int64) = _simulate(m, start, steps); 
(m::MyHiddenMarkovModelWithJumps)(start::Int64, steps::Int64) = _simulate(m, start, steps); 
(m::MyHiddenMarkovModelWithJumps)(start::Int64, steps::Int64) = _simulate_pagerank(m, start, steps); 


"""
    learn_return_distribution_mcmc(returns::Vector{Float64}; samples::Int = 2000)

Uses a Bayesian MCMC approach to learn the parameters of a Student's t-distribution
fitted to the equity returns data.

Returns a Turing.jl `Chain` object containing the posterior distributions of the parameters.
"""
function learn_distribution_mcmc(model_type::AbstractDistributionModel, returns::Vector{Float64}; samples::Int = 2000)
    
    # 1. Build the correct model based on the input type (e.g., StudentTModel())
    #    Julia's multiple dispatch calls the correct function from Factory.jl
    model_instance = build_turing_model(model_type, returns)

    # 2. Run the MCMC sampler
    chain = Turing.sample(model_instance, NUTS(), samples)

    # 3. Return the resulting chain
    return chain
end

# ========================================================================================= #
# Growth Calculation Functions
# ========================================================================================= #

"""
    log_growth_matrix(dataset::Dict{String, DataFrame}, firms::Array{String,1}; ...)

Computes the excess log growth matrix for a list of firms.
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
    log_growth_matrix(dataset::Dict{String, DataFrame}, firm::String; ...)

Computes the excess log growth vector for a single firm (by String ticker).
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

Computes the excess log growth vector for a single DataFrame.
"""
function log_growth_matrix(dataset::DataFrame; 
    Δt::Float64 = (1.0/252.0), risk_free_rate::Float64 = 0.0,
    keycol::Symbol = :volume_weighted_average_price)::Array{Float64,1}

    # initialize -
    firm_data = dropmissing(dataset, disallowmissing=true)
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

Computes the excess log growth vector for a raw array of prices.
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


"""
    boost_persistence!(m::MyHiddenMarkovModelWithJumps, persistence_factor::Float64)

Modifies the transition matrix in-place to increase the 'stickiness' of tail states.
This generates volatility clustering without artificial duration counters.

- `persistence_factor`: A value between 0.0 and 1.0. 
   Higher values force the tail states to be more sticky (higher self-transition).
"""
function boost_persistence!(m::MyHiddenMarkovModelWithJumps, persistence_factor::Float64)
    
    n_states = length(m.states)
    # Identify tail states (modify these indices based on your sort logic)
    # Assuming sorted by return: 1-3 are Crash, (N-2)-N are Boom
    tail_indices = union(1:3, (n_states-2):n_states)

    for s in tail_indices
        # Get current probabilities
        probs = m.transition[s].p
        
        # 1. Boost the diagonal (Self-Transition)
        # We blend the current diagonal with 1.0 based on the factor
        current_diag = probs[s]
        new_diag = current_diag + (1.0 - current_diag) * persistence_factor
        
        # 2. Renormalize the off-diagonal elements
        # They must sum to (1 - new_diag)
        off_diagonal_sum = sum(probs) - current_diag
        
        if off_diagonal_sum > 0
            scaling_ratio = (1.0 - new_diag) / off_diagonal_sum
            
            # Apply scaling to all, then overwrite diagonal
            new_probs = probs .* scaling_ratio
            new_probs[s] = new_diag
            
            # Update the model's transition matrix
            m.transition[s] = Categorical(new_probs)
        else
            # If off-diagonal was already 0, just ensure diagonal is 1.0
            new_probs = zeros(length(probs))
            new_probs[s] = 1.0
            m.transition[s] = Categorical(new_probs)
        end
    end
    println("Tail state persistence boosted by factor $(persistence_factor)")
end