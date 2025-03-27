# adaptive_model.jl
using Pkg
Pkg.add(["CSV", "DataFrames", "Statistics", "BenchmarkTools", "CPUTime", "TimerOutputs", "JSON", "Plots", "LIBSVM", "DecisionTree", "Flux", "GLM", "XGBoost", "Printf", "LinearAlgebra"])

using CSV, DataFrames, Statistics
using BenchmarkTools, TimerOutputs, CPUTime
using JSON, Plots, Random
using LIBSVM, DecisionTree, Flux, GLM, XGBoost
using Printf
using Dates
using LinearAlgebra

# structures 
struct AttackPattern
    features::Vector{Float64}
    timestamp::DateTime
    success_rate::Float64
end

mutable struct PatternEvolution
    known_patterns::Dict{String,Vector{AttackPattern}}
    similarity_threshold::Float64
    pattern_lifetime::Float64  # in hours
    adaptation_rate::Float64
end

# Create a timer
const to = TimerOutput()

# Load data and split into phases
data = CSV.read("LR-HR DDoS 2024 Dataset for SDN-Based Networks.csv", DataFrame)

# Select features
selected_features = [
    "Flow Pkts/s", "SYN Flag Cnt", "packet_count", "flow_duration",
    "RST Flag Cnt", "byte_count", "FIN Flag Cnt", "Pkt Size Avg", "Label"
]

# Prepare data
X = Matrix(data[:, selected_features[1:end-1]])
y = Vector(data[:, "Label"])

# Normalize features
feature_means = mean(X, dims=1)
feature_stds = std(X, dims=1)
X_normalized = (X .- feature_means) ./ feature_stds

# Split data
Random.seed!(42)
n = size(X_normalized, 1)
train_size = Int(floor(0.6 * n))
adapt_size = Int(floor(0.2 * n))
indices = shuffle(1:n)

train_idx = indices[1:train_size]
adapt_idx = indices[train_size+1:train_size+adapt_size]
test_idx = indices[train_size+adapt_size+1:end]

X_train = X_normalized[train_idx, :]
X_adapt = X_normalized[adapt_idx, :]
X_test = X_normalized[test_idx, :]

y_train = y[train_idx]
y_adapt = y[adapt_idx]
y_test = y[test_idx]

# Train individual models
function train_models()
    println("\n=== Training Individual Models ===")

    # SVM
    println("Training SVM...")
    svm_time = @elapsed begin
        svm_model = svmtrain(transpose(X_train), y_train)
    end
    println("SVM Training Time: $(round(svm_time, digits=3)) seconds")

    # Random Forest
    println("\nTraining Random Forest...")
    rf_time = @elapsed begin
        rf_model = RandomForestClassifier(n_trees=100)
        DecisionTree.fit!(rf_model, X_train, y_train)
    end
    println("RF Training Time: $(round(rf_time, digits=3)) seconds")

    # Logistic Regression
    println("\nTraining Logistic Regression...")
    lr_time = @elapsed begin
        lr_model = glm(X_train, y_train, Binomial(), LogitLink())
    end
    println("LR Training Time: $(round(lr_time, digits=3)) seconds")

    # XGBoost
    println("\nTraining XGBoost...")
    xgb_time = @elapsed begin
        # Convert labels to Float32
        y_train_xgb = Float32.(y_train)

        # Create DMatrix
        dtrain = DMatrix(X_train, y_train_xgb)

        # Set parameters
        params = Dict(
            "objective" => "binary:logistic",
            "eval_metric" => "error",
            "eta" => 0.1,
            "max_depth" => 6
        )

        # Train model
        xgb_model = xgboost(dtrain;
            num_round=100,
            params=params,
            watchlist=Dict("train" => dtrain)
        )
    end
    println("XGB Training Time: $(round(xgb_time, digits=3)) seconds")

    return Dict(
        "SVM" => svm_model,
        "RF" => rf_model,
        "LR" => lr_model,
        "XGB" => xgb_model
    )
end

# Get predictions from models
function get_predictions(models, X)
    predictions = Dict()

    # Use views and preallocate where possible
    X_t = transpose(X)  # Single transpose operation

    # SVM predictions
    svm_pred, _ = svmpredict(models["SVM"], X_t)
    predictions["SVM"] = svm_pred

    # RF predictions (using view)
    rf_pred = DecisionTree.predict(models["RF"], X)
    predictions["RF"] = rf_pred

    # LR predictions (minimize allocations)
    lr_probs = GLM.predict(models["LR"], X)
    predictions["LR"] = Float64.(lr_probs .> 0.5)

    # XGBoost predictions (reuse DMatrix if possible)
    dtest = DMatrix(X)
    predictions["XGB"] = Float64.(XGBoost.predict(models["XGB"], dtest) .> 0.5)

    return predictions
end

# Adaptive ensemble function
function adaptive_ensemble(predictions, weights)
    weighted_pred = sum(pred .* weights[model] for (model, pred) in predictions)
    return weighted_pred .> 0.5
end

# Performance measurement function
function measure_performance(f)
    # Get initial memory state
    mem_before = Sys.total_memory() - Sys.free_memory()

    # Get initial CPU time
    cpu_time_before = CPUTime.CPUtime_us()

    # Time and run function
    time = @elapsed begin
        @timeit to "execution" result = f()
    end

    # Get final memory state
    mem_after = Sys.total_memory() - Sys.free_memory()
    mem_used = (mem_after - mem_before) / 1024 / 1024  # Convert to MB

    # Get final CPU time
    cpu_time_after = CPUTime.CPUtime_us()
    cpu_time_used = (cpu_time_after - cpu_time_before) / 1e6  # Convert to seconds

    return result, time, mem_used, cpu_time_used
end

# Add these functions before "# Main testing function" line
function cosine_similarity(a::Vector{Float64}, b::Vector{Float64})
    return dot(a, b) / (norm(a) * norm(b))
end

function detect_and_adapt!(pattern_evolution, new_features, attack_type, success_rate)
    current_time = now()
    new_pattern = AttackPattern(
        vec(new_features),
        current_time,
        success_rate
    )

    # Check if this is a new pattern variation
    is_new_pattern = true
    if haskey(pattern_evolution.known_patterns, attack_type)
        for known_pattern in pattern_evolution.known_patterns[attack_type]
            similarity = cosine_similarity(new_pattern.features, known_pattern.features)
            if similarity > pattern_evolution.similarity_threshold
                is_new_pattern = false
                break
            end
        end
    end

    # If new pattern detected, store it
    if is_new_pattern
        println("New attack pattern variation detected for: $attack_type")
        if !haskey(pattern_evolution.known_patterns, attack_type)
            pattern_evolution.known_patterns[attack_type] = AttackPattern[]
        end
        push!(pattern_evolution.known_patterns[attack_type], new_pattern)
    end

    cleanup_old_patterns!(pattern_evolution, current_time)
end

function adaptive_testing_loop(models, X_test, y_test, pattern_evolution)
    batch_size = 1000  # Can be adjusted based on available memory
    n_samples = size(X_test, 1)
    predictions = Dict(model => Float64[] for model in keys(models))

    for start_idx in 1:batch_size:n_samples
        end_idx = min(start_idx + batch_size - 1, n_samples)

        # Process batch
        X_batch = view(X_test, start_idx:end_idx, :)  # Use view instead of copy
        y_batch = view(y_test, start_idx:end_idx)     # Use view instead of copy

        batch_predictions = get_predictions(models, X_batch)

        # Store predictions and immediately release batch memory
        for (model, preds) in batch_predictions
            append!(predictions[model], preds)
        end

        GC.gc()  # Force garbage collection after each batch
    end
    return predictions
end

function cleanup_old_patterns!(pattern_evolution, current_time)
    for (attack_type, patterns) in pattern_evolution.known_patterns
        filter!(p -> (current_time - p.timestamp).value / 3600 < pattern_evolution.pattern_lifetime,
            pattern_evolution.known_patterns[attack_type])
    end
end

function update_weights_for_pattern!(models, weights, performances)
    total_perf = sum(values(performances))
    if total_perf > 0
        for model in keys(weights)
            weights[model] = performances[model] / total_perf
        end
    end
end

# Move this function up, after the other function definitions (around line 150, before the main testing function)
function calculate_ensemble_metrics(y_true, y_pred)
    # Convert predictions to same type as true labels
    y_pred_int = Int.(y_pred)

    # Calculate confusion matrix elements
    tp = sum((y_true .== 1) .& (y_pred_int .== 1))
    tn = sum((y_true .== 0) .& (y_pred_int .== 0))
    fp = sum((y_true .== 0) .& (y_pred_int .== 1))
    fn = sum((y_true .== 1) .& (y_pred_int .== 0))

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1_score
end

# Add these functions for advanced pattern analysis and adaptation
function extract_attack_features(X_batch, y_batch)
    # Extract features only from attack samples
    attack_indices = findall(y_batch .== 1)
    return X_batch[attack_indices, :]
end

function cluster_attack_patterns(features, n_clusters=3)
    # Simple k-means implementation for attack pattern clustering
    # In a real implementation, you might want to use a dedicated clustering package
    n_samples = size(features, 1)
    if n_samples == 0
        return Int[], Vector{Vector{Float64}}()
    end

    # Initialize centroids randomly
    centroids = features[rand(1:n_samples, n_clusters), :]

    # Assign points to clusters
    assignments = zeros(Int, n_samples)
    for iter in 1:10  # Limited iterations for efficiency
        # Assign each point to nearest centroid
        for i in 1:n_samples
            distances = [norm(features[i, :] - centroids[j, :]) for j in 1:n_clusters]
            assignments[i] = argmin(distances)
        end

        # Update centroids
        new_centroids = zeros(size(centroids))
        counts = zeros(Int, n_clusters)
        for i in 1:n_samples
            new_centroids[assignments[i], :] .+= features[i, :]
            counts[assignments[i]] += 1
        end

        # Avoid division by zero
        for j in 1:n_clusters
            if counts[j] > 0
                new_centroids[j, :] ./= counts[j]
            else
                # If a cluster is empty, reinitialize its centroid
                new_centroids[j, :] = features[rand(1:n_samples), :]
            end
        end

        centroids = new_centroids
    end

    return assignments, [centroids[i, :] for i in 1:n_clusters]
end

function detect_concept_drift(historical_performance, current_performance, window_size=10, threshold=0.05)
    if length(historical_performance) < window_size
        return false
    end

    avg_historical = mean(historical_performance[end-window_size+1:end])
    drift_detected = abs(current_performance - avg_historical) > threshold

    return drift_detected
end

function adaptive_weight_adjustment(weights, performances, adaptation_rate, drift_detected)
    # Increase adaptation rate if drift is detected
    effective_rate = drift_detected ? adaptation_rate * 2.0 : adaptation_rate

    # Calculate new weights based on recent performance
    total_perf = sum(values(performances))
    new_weights = Dict()

    if total_perf > 0
        for model in keys(weights)
            # Blend old weights with new performance-based weights
            new_weights[model] = (1 - effective_rate) * weights[model] +
                                 effective_rate * (performances[model] / total_perf)
        end
    else
        new_weights = weights  # Keep old weights if no performance data
    end

    return new_weights
end

function feature_importance_analysis(models, X, y)
    importances = Dict()

    # For Random Forest, we need to implement our own feature importance calculation
    # since DecisionTree.feature_importance doesn't exist
    if haskey(models, "RF")
        # Use permutation importance approach for RF as well
        rf_model = models["RF"]
        baseline_preds = DecisionTree.predict(rf_model, X)
        baseline_acc = mean(baseline_preds .== y)

        rf_importances = zeros(size(X, 2))
        for feat_idx in 1:size(X, 2)
            # Create permuted data
            X_permuted = copy(X)
            X_permuted[:, feat_idx] = X_permuted[shuffle(1:size(X, 1)), feat_idx]

            # Get predictions with permuted feature
            perm_preds = DecisionTree.predict(rf_model, X_permuted)
            perm_acc = mean(perm_preds .== y)

            # Importance is the decrease in performance
            rf_importances[feat_idx] = baseline_acc - perm_acc
        end

        importances["RF"] = rf_importances
    end

    # For other models, we can use a permutation importance approach
    for model_name in ["SVM", "LR", "XGB"]
        if haskey(models, model_name)
            # Get baseline performance
            preds = model_name == "SVM" ? svmpredict(models[model_name], transpose(X))[1] :
                    model_name == "LR" ? Float64.(GLM.predict(models[model_name], X) .> 0.5) :
                    Float64.(XGBoost.predict(models[model_name], DMatrix(X)) .> 0.5)

            baseline_acc = mean(preds .== y)

            # Calculate importance for each feature
            feature_imps = zeros(size(X, 2))
            for feat_idx in 1:size(X, 2)
                # Create permuted data
                X_permuted = copy(X)
                X_permuted[:, feat_idx] = X_permuted[shuffle(1:size(X, 1)), feat_idx]

                # Get predictions with permuted feature
                perm_preds = model_name == "SVM" ? svmpredict(models[model_name], transpose(X_permuted))[1] :
                             model_name == "LR" ? Float64.(GLM.predict(models[model_name], X_permuted) .> 0.5) :
                             Float64.(XGBoost.predict(models[model_name], DMatrix(X_permuted)) .> 0.5)

                perm_acc = mean(perm_preds .== y)

                # Importance is the decrease in performance
                feature_imps[feat_idx] = baseline_acc - perm_acc
            end

            importances[model_name] = feature_imps
        end
    end

    return importances
end

# Enhanced adaptive ensemble function with confidence scores
function adaptive_ensemble_with_confidence(predictions, weights, X)
    # Calculate weighted prediction
    weighted_pred = sum(pred .* weights[model] for (model, pred) in predictions)

    # Calculate confidence scores (distance from decision boundary)
    confidence_scores = abs.(weighted_pred .- 0.5) .* 2  # Scale to [0,1]

    # Final predictions
    final_pred = weighted_pred .> 0.5

    return final_pred, confidence_scores
end

# Enhanced adaptive testing loop with concept drift detection
function enhanced_adaptive_testing_loop(models, X_test, y_test, pattern_evolution)
    batch_size = 1000
    n_samples = size(X_test, 1)
    predictions = Dict(model => Float64[] for model in keys(models))

    # Track performance over time for drift detection
    historical_performance = Float64[]
    drift_detected_points = Int[]

    # Initialize weights
    weights = Dict(model => 1.0 / length(models) for model in keys(models))

    for start_idx in 1:batch_size:n_samples
        end_idx = min(start_idx + batch_size - 1, n_samples)
        batch_range = start_idx:end_idx

        # Process batch
        X_batch = view(X_test, batch_range, :)
        y_batch = view(y_test, batch_range)

        # Get predictions for this batch
        batch_predictions = get_predictions(models, X_batch)

        # Calculate performance for each model on this batch
        batch_performances = Dict()
        for (model, preds) in batch_predictions
            batch_performances[model] = mean(preds .== y_batch)
        end

        # Check for concept drift
        current_ensemble_perf = mean(adaptive_ensemble(batch_predictions, weights) .== y_batch)
        push!(historical_performance, current_ensemble_perf)

        drift_detected = detect_concept_drift(historical_performance, current_ensemble_perf)
        if drift_detected
            push!(drift_detected_points, start_idx)
            println("Concept drift detected at sample $(start_idx)")

            # Extract attack patterns for adaptation
            attack_features = extract_attack_features(X_batch, y_batch)
            if size(attack_features, 1) > 0
                # Cluster attack patterns
                _, centroids = cluster_attack_patterns(attack_features)

                # Register new patterns
                for centroid in centroids
                    detect_and_adapt!(pattern_evolution, centroid, "adaptive_attack",
                        1.0 - current_ensemble_perf)
                end
            end
        end

        # Adapt weights based on recent performance
        weights = adaptive_weight_adjustment(weights, batch_performances,
            pattern_evolution.adaptation_rate, drift_detected)

        # Store predictions
        for (model, preds) in batch_predictions
            append!(predictions[model], preds)
        end

        GC.gc()  # Force garbage collection after each batch
    end

    return predictions, weights, historical_performance, drift_detected_points
end

# Add explainability function
function generate_explanation(models, X_sample, feature_names, importances)
    # Get predictions from each model
    sample_predictions = Dict()
    for model_name in keys(models)
        if model_name == "SVM"
            sample_predictions[model_name] = svmpredict(models[model_name], reshape(X_sample, 1, length(X_sample))')[1][1]
        elseif model_name == "RF"
            sample_predictions[model_name] = DecisionTree.predict(models[model_name], reshape(X_sample, 1, length(X_sample)))[1]
        elseif model_name == "LR"
            sample_predictions[model_name] = GLM.predict(models[model_name], reshape(X_sample, 1, length(X_sample)))[1] > 0.5 ? 1.0 : 0.0
        elseif model_name == "XGB"
            sample_predictions[model_name] = XGBoost.predict(models[model_name], DMatrix(reshape(X_sample, 1, length(X_sample))))[1] > 0.5 ? 1.0 : 0.0
        end
    end

    # Find top contributing features for each model
    explanations = Dict()
    for model_name in keys(importances)
        model_imp = importances[model_name]
        # Get indices of top 3 features
        top_features_idx = sortperm(model_imp, rev=true)[1:min(3, length(model_imp))]

        # Create explanation
        explanations[model_name] = Dict(
            "prediction" => sample_predictions[model_name],
            "top_features" => [(feature_names[idx], model_imp[idx]) for idx in top_features_idx]
        )
    end

    return explanations
end

# Main testing function
println("\n=== Adaptive Ensemble Testing ===")
ensemble_result, ensemble_time, ensemble_memory, ensemble_cpu = measure_performance() do
    GC.gc()  # Clean up before testing

    # Train models with memory optimization
    println("Training individual models...")
    models = train_models()

    # Initial weights
    weights = Dict(
        "SVM" => 0.25,
        "RF" => 0.25,
        "LR" => 0.25,
        "XGB" => 0.25
    )

    # Track performance
    performances = Dict("SVM" => 0.0, "RF" => 0.0, "LR" => 0.0, "XGB" => 0.0)

    # Adaptation phase
    println("Starting adaptation phase...")
    predictions = get_predictions(models, X_adapt)
    for (model, preds) in predictions
        accuracy = mean(preds .== y_adapt)
        performances[model] = accuracy
    end

    # Calculate feature importance
    println("Analyzing feature importance...")
    feature_importances = feature_importance_analysis(models, X_adapt, y_adapt)

    # Update weights based on adaptation performance
    update_weights_for_pattern!(models, weights, performances)

    # Initialize pattern evolution with more aggressive adaptation
    pattern_evolution = PatternEvolution(
        Dict{String,Vector{AttackPattern}}(),
        0.80,  # Lower similarity threshold to detect more pattern variations
        48.0,  # Longer pattern lifetime in hours
        0.15   # Higher adaptation rate for faster response
    )

    # Get test predictions with enhanced adaptive loop
    println("Starting adaptive test phase...")
    test_predictions, final_weights, performance_history, drift_points =
        enhanced_adaptive_testing_loop(models, X_test, y_test, pattern_evolution)

    # Calculate ensemble predictions with confidence
    ensemble_preds, confidence_scores = adaptive_ensemble_with_confidence(test_predictions, final_weights, X_test)

    # Generate explanations for a few test samples
    sample_explanations = Dict()
    if length(y_test) > 0
        # Select a few samples of each class for explanation
        normal_samples = findall(y_test .== 0)[1:min(3, count(y_test .== 0))]
        attack_samples = findall(y_test .== 1)[1:min(3, count(y_test .== 1))]

        for idx in vcat(normal_samples, attack_samples)
            sample_explanations[idx] = generate_explanation(
                models,
                X_test[idx, :],
                selected_features[1:end-1],
                feature_importances
            )
        end
    end

    return Dict(
        "predictions" => ensemble_preds,
        "confidence" => confidence_scores,
        "weights" => final_weights,
        "individual_performances" => performances,
        "test_predictions" => test_predictions,
        "performance_history" => performance_history,
        "drift_points" => drift_points,
        "feature_importances" => feature_importances,
        "sample_explanations" => sample_explanations
    )
end

# Calculate and print comprehensive metrics
println("\n=== Performance Metrics ===")

# Calculate metrics for ensemble predictions
ensemble_preds = ensemble_result["predictions"]
ensemble_acc, ensemble_prec, ensemble_rec, ensemble_f1 = calculate_ensemble_metrics(y_test, ensemble_preds)

println("Adaptive Ensemble Performance:")
println("  Accuracy:  $(round(ensemble_acc * 100, digits=2))%")
println("  Precision: $(round(ensemble_prec * 100, digits=2))%")
println("  Recall:    $(round(ensemble_rec * 100, digits=2))%")
println("  F1 Score:  $(round(ensemble_f1 * 100, digits=2))%")

# Calculate metrics for individual models
println("\nIndividual Model Performance:")
for (model, preds) in ensemble_result["test_predictions"]
    acc, prec, rec, f1 = calculate_ensemble_metrics(y_test, preds .> 0.5)
    println("  $model:")
    println("    Accuracy:  $(round(acc * 100, digits=2))%")
    println("    Precision: $(round(prec * 100, digits=2))%")
    println("    Recall:    $(round(rec * 100, digits=2))%")
    println("    F1 Score:  $(round(f1 * 100, digits=2))%")
end

# Print resource utilization metrics
println("\n=== Resource Utilization ===")
println("Execution Time:     $(round(ensemble_time, digits=3)) seconds")
println("Memory Usage:       $(round(ensemble_memory, digits=2)) MB")
println("CPU Time:           $(round(ensemble_cpu, digits=3)) seconds")

# Calculate efficiency metrics
efficiency_score = ensemble_acc / (ensemble_time * 0.01)  # Normalize time impact
println("Efficiency Score:   $(round(efficiency_score, digits=2)) (accuracy/time)")

# Print final model weights
println("\n=== Final Model Weights ===")
for (model, weight) in ensemble_result["weights"]
    println("  $model: $(round(weight * 100, digits=2))%")
end

# Add detailed resource tracking for each phase
println("\n=== Detailed Resource Tracking ===")
println("Timer Output Summary:")
show(to)
println()

# Create a comprehensive performance report
performance_report = Dict(
    "accuracy" => ensemble_acc,
    "precision" => ensemble_prec,
    "recall" => ensemble_rec,
    "f1_score" => ensemble_f1,
    "execution_time" => ensemble_time,
    "memory_usage" => ensemble_memory,
    "cpu_time" => ensemble_cpu,
    "efficiency_score" => efficiency_score,
    "model_weights" => ensemble_result["weights"],
    "drift_points_detected" => length(ensemble_result["drift_points"]),
    "timestamp" => string(now())
)

# Save performance report to JSON
open("performance_report.json", "w") do f
    JSON.print(f, performance_report, 4)  # Pretty print with 4-space indent
end
println("\nPerformance report saved to performance_report.json")

# Create a confusion matrix visualization
if length(y_test) > 0
    # Calculate confusion matrix
    tp = sum((y_test .== 1) .& (ensemble_preds .== 1))
    tn = sum((y_test .== 0) .& (ensemble_preds .== 0))
    fp = sum((y_test .== 0) .& (ensemble_preds .== 1))
    fn = sum((y_test .== 1) .& (ensemble_preds .== 0))

    # Create confusion matrix plot
    cm = [tn fp; fn tp]
    p_cm = heatmap(["Predicted Negative", "Predicted Positive"],
        ["Actual Negative", "Actual Positive"],
        cm,
        color=:blues,
        title="Confusion Matrix",
        fmt=:d)

    # Add text annotations to the heatmap
    annotate!(p_cm, [(1, 1, text("TN: $tn", 10, :white)),
        (2, 1, text("FP: $fp", 10, :white)),
        (1, 2, text("FN: $fn", 10, :white)),
        (2, 2, text("TP: $tp", 10, :white))])

    savefig(p_cm, "confusion_matrix.png")
    display(p_cm)
end