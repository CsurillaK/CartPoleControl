environment = CartPoleEnvironment();
environment.Mode = 2; % Unstable

%% Build agent
unstableAgent = BuildCartPoleAgent(environment, [350, 200], [250, 200]);

%% Train
environment.TrajectoryCollection.Parameter.Unstable.StableStartingPositionProbability = 0.7;
environment.TrajectoryCollection.Parameter.Unstable.StandstillProbability = 0.1;

environment.TrajectoryCollection.Parameter.Unstable.UnstableStartingAngleDistributionWidth = pi;

environment.Reward.Parameter.Unstable.PenaltyOutOfBoundary = 0;
environment.Reward.Parameter.Unstable.RewardWithinBoundary = 50;
environment.Reward.Parameter.Unstable.Gainx1 = 3;
environment.Reward.Parameter.Unstable.Gainy1 = 3;
environment.Reward.Parameter.Unstable.Gainx0d = 0.6;
environment.Reward.Parameter.Unstable.Gainx1d = 0.6;
environment.Reward.Parameter.Unstable.GainF = 0.1;

unstableAgent.AgentOptions.CriticOptimizerOptions.LearnRate = 1e-5;
unstableAgent.AgentOptions.ActorOptimizerOptions.LearnRate = 1e-5;

unstableAgent.AgentOptions.MiniBatchSize = 256;
unstableAgent.AgentOptions.NoiseOptions.Variance = 0.5; % 0.4
unstableAgent.AgentOptions.NoiseOptions.VarianceDecayRate = 0.02;
unstableAgent.AgentOptions.DiscountFactor = 0.995;

maxEpisodes = 200;
trainOptions = rlTrainingOptions( ...
    MaxEpisodes = maxEpisodes, ...
    MaxStepsPerEpisode = 300, ... 5 / environment.Physics.Ts
    ScoreAveragingWindowLength = 5, ...
    StopTrainingCriteria = "EpisodeCount", ...
    StopTrainingValue = maxEpisodes);
trainingStats = train(unstableAgent, environment, trainOptions);

%% Test agent
environment.InteractiveAgent(unstableAgent);