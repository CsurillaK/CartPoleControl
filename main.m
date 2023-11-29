addpath(".\Misc");

%%
environment = CartPole.Environment();

load("Agents/Stable/agent_4.mat");
load("Agents/Unstable/agent_4.mat");

%%
environment.InteractiveDoubleAgent(stableAgent, unstableAgent);
