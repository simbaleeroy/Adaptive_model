This is a project meant to demonstrate the effectiveness of an adaptive ensemble model on detecting DDoS attacks.

The model has four key innovative components: • Base Model Ensemble: Four different machine learning algorithms are used (SVM, Random Forest, Logistic Regression, and XGBoost). Initially equal weights are given to each model and are increased and reduced for the best and least performing model respectively during training iterations. • Pattern Evolution Tracking: The architecture consists of a memory system that contains known attack patterns and uses the cosine similarity to detect new attacks. • Drift Detection: We keep track of model performance using the sliding window technique. When there is a drift, rates of adaptation are incremented automatically, and new patterns are added to the memory for future use. • Explainability Framework: Feature importance score is determined for each prediction, giving insights into model decision making.

The model is trained on a dataset of over 100 000 records of network flows and tested on its effectiveness in a simulated environment. The results of model performance are shown below:

Performance and Resource Usage Metrics

Accuracy 99.98% Precision 100% Recall 99.96% F1 score 99.98% Memory 10.06MB CPU Usage 105.20MB


@GET
@Timed(name = "listAutoNetConfigs")
@Produces({"application/json", "application/x-json-stream"})
@Path("/autoNetConfigs")
@VpnSkipAuthorization
public PaginatedResponse<AutoNetConfig> listAutoNetConfigs(
        @DefaultValue(LIST_LIMIT) @QueryParam(Params.LIMIT) int limit,
        @QueryParam(Params.PAGE) String opaqueToken,
        @QueryParam(CustomParams.CONFIG_REFERENCE_ID) String referenceId,
        @QueryParam(CustomParams.CONFIG_DEVICE_ID) String deviceId,
        @QueryParam(CustomParams.CONFIG_STATE) String state) throws ValidationException {
    AutoNetConfigDo.State queryState = state != null ? AutoNetConfigDo.State.valueOf(state) : null;
    Predicate<AutoNetConfigDo> filterByDevice = autoNetConfigDo -> ((deviceId == null || autoNetConfigDo.getDeviceId().equals(deviceId)));
    Predicate<AutoNetConfigDo> filterByState = autoNetConfigDo -> ((queryState == null || autoNetConfigDo.getState() == queryState));
    Predicate<AutoNetConfigDo> filterByReferenceId = autoNetConfigDo -> ((referenceId == null || autoNetConfigDo.getReferenceId().equals(referenceId)));

    if (referenceId != null) {
        return new PaginatedResponse<>(autoNetConfigBl.listSnippetsForDomainObject(referenceId).stream()
                .filter(filterByDevice).filter(filterByState)
                .map(ManagementResourceV1::toModel).collect(Collectors.toList()));
    } else if (deviceId != null) {
        return PaginationUtils.getScanPage(
                opaqueToken,
                () -> autoNetConfigBl.listSnippetsForDevice(deviceId, Optional.empty(), limit),
                (PaginationToken p) -> autoNetConfigBl.listSnippetsForDevice(
                        deviceId, PaginationTokenUtils.deserializeAndValidate(opaqueToken, autoNetConfigBl.getMappedDataStore()), limit),
                filterByState.and(filterByReferenceId),
                ManagementResourceV1::toModel,
                autoNetConfigBl.getMappedDataStore()
        );
    } else if (queryState != null) {
        return PaginationUtils.getScanPage(
                opaqueToken,
                () -> autoNetConfigBl.listSnippetsForState(queryState, Optional.empty(), limit),
                (PaginationToken p) -> autoNetConfigBl.listSnippetsForState(
                        queryState, PaginationTokenUtils.deserializeAndValidate(opaqueToken, autoNetConfigBl.getMappedDataStore()), limit),
                filterByReferenceId.and(filterByDevice),
                ManagementResourceV1::toModel,
                autoNetConfigBl.getMappedDataStore()
        );
    } else {
        return PaginationUtils.getScanPage(opaqueToken, limit, autoNetConfigBl, ManagementResourceV1::toModel);
    }
}

public static AutoNetConfig toModel(AutoNetConfigDo autoNetConfigDo) {
    return AutoNetConfig.builder()
            .id(autoNetConfigDo.getId())
            .deviceId(autoNetConfigDo.getDeviceId())
            .referenceId(autoNetConfigDo.getReferenceId())
            .isDrainRequired(autoNetConfigDo.getIsDrainRequired())
            .config(autoNetConfigDo.getConfig() != null ? autoNetConfigDo.getConfig().toString() : null)
            .lastAttempt(autoNetConfigDo.getLastAttempt())
            .attempts(autoNetConfigDo.getAttempts())
            .failed(autoNetConfigDo.getFailed())
            .dateCreated(autoNetConfigDo.getDateCreated())
            .lastUpdated(autoNetConfigDo.getLastUpdated())
            .stateComment(autoNetConfigDo.getStateComment())
            .changeTicket(autoNetConfigDo.getChangeTicket())
            .state(AutoNetConfig.State.valueOf(autoNetConfigDo.getState().toString()))
            .build();
}
