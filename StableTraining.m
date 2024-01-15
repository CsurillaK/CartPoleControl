environment = CartPole.Environment();
environment.Mode = environment.ModeStable; % Stable

%% Initialize agent
stableAgent = BuildCartPoleAgent(environment, [300, 200], [200, 100]); % [200, 100], [100, 50]

%% Train
environment.Trajectory.Parameter.Stable.UnstableStartingPositionProbability = 0.2; % 0.1
environment.Trajectory.Parameter.Stable.StandstillProbability = 0.0; % 0.2

environment.Reward.Parameter.Stable.PenaltyOutOfBoundary = 0;
environment.Reward.Parameter.Stable.RewardWithinBoundary = 10;
environment.Reward.Parameter.Stable.Gainx1 = 3;
environment.Reward.Parameter.Stable.Gainy1 = 1;
environment.Reward.Parameter.Stable.Gainx0d = 0.2; % 0.2
environment.Reward.Parameter.Stable.Gainx1d = 0.2; % 0.2
environment.Reward.Parameter.Stable.GainF = 0.1;

stableAgent.AgentOptions.CriticOptimizerOptions.LearnRate = 1e-4; %1e-5;
stableAgent.AgentOptions.ActorOptimizerOptions.LearnRate = 1e-4; %1e-4;

stableAgent.AgentOptions.MiniBatchSize = 256;
stableAgent.AgentOptions.DiscountFactor = 0.99; % 0.98
stableAgent.AgentOptions.ExperienceBufferLength = 200000;

stableAgent.AgentOptions.NoiseOptions = rl.option.GaussianActionNoise();
stableAgent.AgentOptions.NoiseOptions.StandardDeviation = 0.05; % 0.4
stableAgent.AgentOptions.NoiseOptions.StandardDeviationMin = 0.05;
stableAgent.AgentOptions.NoiseOptions.StandardDeviationDecayRate = 0.00005;

% stableAgent.AgentOptions.NoiseOptions = rl.option.OrnsteinUhlenbeckActionNoise();
% stableAgent.AgentOptions.NoiseOptions.SampleTime = 0.05;
% stableAgent.AgentOptions.NoiseOptions.StandardDeviation = 0.01; % 0.3
% stableAgent.AgentOptions.NoiseOptions.StandardDeviationDecayRate = 0.0001;
% stableAgent.AgentOptions.NoiseOptions.StandardDeviationMin = 0.01;
% stableAgent.AgentOptions.NoiseOptions.MeanAttractionConstant = 0.2;

maxEpisodes = 300;
trainOptions = rlTrainingOptions( ...
    MaxEpisodes = maxEpisodes, ...
    MaxStepsPerEpisode = 250, ... 5 / environment.Physics.Ts
    ScoreAveragingWindowLength = 10, ...
    StopTrainingCriteria = "EpisodeCount", ...
    StopTrainingValue = maxEpisodes);
trainingStats = train(stableAgent, environment, trainOptions);

%% Test agent
environment.InteractiveAgent(stableAgent);
