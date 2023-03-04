def main_trainANN(info):
    messages = []
    messages.append(70 * "*" + "\n")
    messages.append(f"Step 2: Train the regressors for all loadings with the initial simulations of curve {CPLaw}{curveIndex}\n\n")
    messages.append(f"ANN model: (parameters) -> (stress values at interpolating strain points)\n")
    
    stringMessage = "ANN configuration:\n"

    logTable = PrettyTable()
    logTable.field_names = ["ANN configurations", "Choice"]

    logTable.add_row(["Number of hidden layers", numberOfHiddenLayers])
    logTable.add_row(["Hidden layer nodes formula", hiddenNodesFormula])
    logTable.add_row(["ANN Optimizer", ANNOptimizer])
    logTable.add_row(["Learning rate", learning_rate])
    logTable.add_row(["L2 regularization term", L2_regularization])

    # The ANN regressors for each loading condition
    regressors = {}
    # The regularization scaler for each loading condition
    scalers = {}

    for loading in loadings:
        logTable.add_row([f"Epochs of {loading}", loading_epochs[CPLaw][loading]])
    
    stringMessage += logTable.get_string()
    stringMessage += "\n\n"

    messages.append(stringMessage)
    
    start = time.time()
    for loading in loadings:
        # All loadings share the same parameters, but different stress values
        paramFeatures = np.array([list(dict(params).values()) for params in list(combined_interpolateCurves[loading].keys())])
        stressLabels = np.array([strainstress["stress"] for strainstress in list(combined_interpolateCurves[loading].values())])

        # Normalizing the data
        scalers[loading] = StandardScaler().fit(paramFeatures[0:initial_length])
        paramFeatures = scalers[loading].transform(paramFeatures)

        # Input and output size of the ANN
        sampleSize = stressLabels.shape[0]
        inputSize = paramFeatures.shape[1]
        outputSize = stressLabels.shape[1]

        messages.append(f"({loading}) paramFeatures shape is {paramFeatures.shape}\n")
        messages.append(f"({loading}) stressLabels shape is {stressLabels.shape}\n")

        regressors[loading] = NeuralNetwork(inputSize, outputSize, hiddenNodesFormula, numberOfHiddenLayers, sampleSize).to(device)
        trainingError = regressors[loading].train(paramFeatures, stressLabels, ANNOptimizer, learning_rate, loading_epochs[CPLaw][loading], L2_regularization)

        messages.append(f"Finish training ANN for loading {loading}\n")
        messages.append(f"Training MSE error: {trainingError[-1]}\n\n")

    end = time.time()

    messages.append(f"The number of combined interpolate curves is {len(combined_interpolateCurves[loading])}\n")
    messages.append(f"Finish training ANN for all loadings of curve {CPLaw}{curveIndex}\n")
    messages.append(f"Total training time: {end - start}s\n\n")

    printList(messages, curveIndex)

if __name__ == '__main__':
    info = optimize_config.main()
    main_prepare(info)