environment = CartPole.Environment();
environment.Mode = 1; % Stable

%% Initialize agent
stableAgent = BuildCartPoleAgent(environment, [300, 200], [200, 100]); % [200, 100], [100, 50]

%% Train
environment.Trajectory.Parameter.Stable.UnstableStartingPositionProbability = 0.1; % 0.1
environment.Trajectory.Parameter.Stable.StandstillProbability = 0.1; % 0.2

environment.Reward.Parameter.Stable.PenaltyOutOfBoundary = 0;
environment.Reward.Parameter.Stable.RewardWithinBoundary = 10;
environment.Reward.Parameter.Stable.Gainx1 = 3;
environment.Reward.Parameter.Stable.Gainy1 = 1;
environment.Reward.Parameter.Stable.Gainx0d = 0.0; % 0.2
environment.Reward.Parameter.Stable.Gainx1d = 0.2; % 0.2
environment.Reward.Parameter.Stable.GainF = 0.1;

stableAgent.AgentOptions.CriticOptimizerOptions.LearnRate = 1e-4; %1e-5;
stableAgent.AgentOptions.ActorOptimizerOptions.LearnRate = 1e-5; %1e-4;

stableAgent.AgentOptions.MiniBatchSize = 256;
stableAgent.AgentOptions.NoiseOptions.Variance = 0.4;
stableAgent.AgentOptions.NoiseOptions.VarianceDecayRate = 0.02;
stableAgent.AgentOptions.DiscountFactor = 0.99; % 0.98

maxEpisodes = 200;
trainOptions = rlTrainingOptions( ...
    MaxEpisodes = maxEpisodes, ...
    MaxStepsPerEpisode = 250, ... 5 / environment.Physics.Ts
    ScoreAveragingWindowLength = 5, ...
    StopTrainingCriteria = "EpisodeCount", ...
    StopTrainingValue = maxEpisodes);
trainingStats = train(stableAgent, environment, trainOptions);

%% Test agent
environment.InteractiveAgent(stableAgent);
