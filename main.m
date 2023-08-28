addpath(".\Misc");

%%
environment = CartPoleEnvironment();

load("Agents/Stable/agent_3.mat");
load("Agents/Unstable/agent_3.mat");

%%
environment.InteractiveDoubleAgent(stableAgent, unstableAgent);