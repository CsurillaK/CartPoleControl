function agent = BuildCartPoleAgent(environment, criticLayerSizes, actorLayerSizes)
observationInfo = getObservationInfo(environment);
actionInfo = getActionInfo(environment);

%% Critic
observationPath = [
    featureInputLayer( ...
        observationInfo.Dimension(1), ...
        Name = "observationPathInputLayer")
    fullyConnectedLayer(criticLayerSizes(1))
    reluLayer()
    fullyConnectedLayer(criticLayerSizes(2), Name = "observationPathOutputLayer")
    ];

actionPath = [
    featureInputLayer( ...
        actionInfo.Dimension(1), ...
        Name="actionPathInputLayer")
    fullyConnectedLayer(criticLayerSizes(2), ...
        Name="actionPathOutputLayer", ...
        BiasLearnRateFactor=0)
    ];
 
commonPath = [
    additionLayer(2, Name="add")
    reluLayer()
    fullyConnectedLayer(1)
    ];

criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork, observationPath);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = addLayers(criticNetwork, commonPath);
criticNetwork = connectLayers(criticNetwork, "observationPathOutputLayer", "add/in1");
criticNetwork = connectLayers(criticNetwork, "actionPathOutputLayer", "add/in2");
criticNetwork = dlnetwork(criticNetwork);
summary(criticNetwork);
analyzeNetwork(criticNetwork);

critic = rlQValueFunction(criticNetwork, ...
    observationInfo, actionInfo, ...
    ObservationInputNames = "observationPathInputLayer", ...
    ActionInputNames = "actionPathInputLayer");

%% Actor
actorNetwork = [
    featureInputLayer(observationInfo.Dimension(1))
    fullyConnectedLayer(actorLayerSizes(1))
    reluLayer()
    fullyConnectedLayer(actorLayerSizes(2))
    reluLayer()
    fullyConnectedLayer(1)
    tanhLayer()
    scalingLayer(Scale = max(actionInfo.UpperLimit))
    ];
actorNetwork = dlnetwork(actorNetwork);
summary(actorNetwork);
analyzeNetwork(actorNetwork);

actor = rlContinuousDeterministicActor(actorNetwork, observationInfo, actionInfo);

%% Agent
criticOptions = rlOptimizerOptions( ...
    LearnRate = 1e-03, ...
    GradientThreshold = 1);
actorOptions = rlOptimizerOptions( ...
    LearnRate = 1e-04, ...
    GradientThreshold = 1);

agentOptions = rlDDPGAgentOptions( ...
    SampleTime = environment.Physics.Ts,...
    CriticOptimizerOptions = criticOptions,...
    ActorOptimizerOptions = actorOptions,...
    ExperienceBufferLength = 1e6,...
    DiscountFactor = 0.95,...
    MiniBatchSize = 128);
agentOptions.NoiseOptions.Variance = 0.6;
agentOptions.NoiseOptions.VarianceDecayRate = 1e-5;

agent = rlDDPGAgent(actor, critic, agentOptions);
end
