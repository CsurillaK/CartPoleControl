environment = CartPole.Environment();
environment.Mode = environment.ModeUnstable; % Unstable

%% Build agent
unstableAgent = BuildCartPoleAgent(environment, [350, 200], [250, 200]);

%% Train
environment.Trajectory.Parameter.Unstable.StableStartingPositionProbability = 0.5;
environment.Trajectory.Parameter.Unstable.StandstillProbability = 0.2;
environment.Trajectory.Parameter.Unstable.StableStartingAngleDistributionWidth = pi;

environment.Reward.Parameter.Unstable.PenaltyOutOfBoundary = 0;
environment.Reward.Parameter.Unstable.RewardWithinBoundary = 10;
environment.Reward.Parameter.Unstable.Gainx1 = 3;
environment.Reward.Parameter.Unstable.Gainy1 = 3;
environment.Reward.Parameter.Unstable.Gainx0d = 1.5; % 0.6
environment.Reward.Parameter.Unstable.Gainx1d = 0.6;
environment.Reward.Parameter.Unstable.GainF = 0.1;

unstableAgent.AgentOptions.CriticOptimizerOptions.LearnRate = 1e-5;
unstableAgent.AgentOptions.ActorOptimizerOptions.LearnRate = 1e-5;

unstableAgent.AgentOptions.MiniBatchSize = 256;
unstableAgent.AgentOptions.DiscountFactor = 0.993;
unstableAgent.AgentOptions.ExperienceBufferLength = 200000;

unstableAgent.AgentOptions.NoiseOptions = rl.option.GaussianActionNoise();
unstableAgent.AgentOptions.NoiseOptions.StandardDeviation = 0.05; % 0.4
unstableAgent.AgentOptions.NoiseOptions.StandardDeviationMin = 0.05;
unstableAgent.AgentOptions.NoiseOptions.StandardDeviationDecayRate = 0.00005;

% unstableAgent.AgentOptions.NoiseOptions = rl.option.OrnsteinUhlenbeckActionNoise();
% unstableAgent.AgentOptions.NoiseOptions.SampleTime = 0.05;
% unstableAgent.AgentOptions.NoiseOptions.StandardDeviation = 0.01; % 0.3
% unstableAgent.AgentOptions.NoiseOptions.StandardDeviationDecayRate = 0.0001;
% unstableAgent.AgentOptions.NoiseOptions.StandardDeviationMin = 0.01;
% unstableAgent.AgentOptions.NoiseOptions.MeanAttractionConstant = 0.2;

maxEpisodes = 300;
trainOptions = rlTrainingOptions( ...
    MaxEpisodes = maxEpisodes, ...
    MaxStepsPerEpisode = 300, ... 5 / environment.Physics.Ts
    ScoreAveragingWindowLength = 10, ...
    StopTrainingCriteria = "EpisodeCount", ...
    StopTrainingValue = maxEpisodes);
trainingStats = train(unstableAgent, environment, trainOptions);

%% Test agent
environment.InteractiveAgent(unstableAgent);