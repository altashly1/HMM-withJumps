function plot_acf_comparison(observed::Vector, simulated::Vector, title_text::String, random_index::Int; is_absolute::Bool=false)

    data_obs = is_absolute ? abs.(observed) : observed
    data_sim = is_absolute ? abs.(simulated) : simulated

    L = 252
    τ = 1:(L-1)
    ci = 1.96 / sqrt(length(data_obs))

    ac_obs = autocor(data_obs, τ)
    ac_sim = autocor(data_sim, τ)
    
    # Change: Added titlefontsize=10
    p = plot(τ, ac_obs, label="Observed", linetype=:steppost, lw=2, c=:red, legend=:topright, title=title_text, titlefontsize=10)
    
    # Change: Updated the label to include the random_index
    plot!(p, τ, ac_sim, label="Simulation (Path $(random_index))", linetype=:steppost, lw=2, c=:blue)
    
    plot!(p, τ, ci * ones(length(τ)), label="95% CI", lw=1.5, c=:gray, ls=:dash)
    plot!(p, τ, -ci * ones(length(τ)), label="", lw=1.5, c=:gray, ls=:dash)
    xlabel!(p, "Lag (trading day)")
    
    # Change: Added "(AU)" to the ylabel
    ylabel!(p, "Autocorrelation (AU)")

    return p
end