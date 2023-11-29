%% Global
padArray = @(inputArray, toLength) [inputArray repmat(inputArray(end), [1, max(toLength - length(inputArray), 0)])];
centralizeEnd = @(inputArray) inputArray - inputArray(end);
unwrap2 = @(inputArray) centralizeEnd(unwrap(inputArray));
%% Akima
akimaPolynomial = makima([-2,-1,0,1,2,3], ...
                          [0,0,0,1,1,1]);

x = 0:0.01:1;
y = ppval(akimaPolynomial, x);

customdiff = @(x) [0 (x(3:end)-x(1:end-2))/2 0];

figure;

subplot(2, 1, 1);
plot(x, y, 'LineWidth', 2);
grid on
xlabel('Normalised time', 'Interpreter', 'latex');
ylabel('Normalised position', 'Interpreter', 'latex');
set(gca, "FontSize", 18, 'TickLabelInterpreter', 'latex')

subplot(2, 1, 2);
plot(x, customdiff(y)/0.01, 'LineWidth', 2);
grid on
xlabel('Normalised time', 'Interpreter', 'latex');
ylabel('Normalised velocity', 'Interpreter', 'latex');
set(gca, "FontSize", 18, 'TickLabelInterpreter', 'latex')
%% Akima transition trajectories
environment = CartPole.Environment();

figure;
Ts = environment.Physics.Ts;
maxLength = environment.Trajectory.Parameter.Stable.TimeLimits(2)/Ts + 1;
for i = 1:10
    [initialObservation, trajectory] = environment.Trajectory.GenerateStable();
    x = 0:Ts:(maxLength - 1) * Ts;
    y = padArray(trajectory(1,:), maxLength);
    plot(x, y, "LineWidth", 2);
    hold on
end

grid on
xlabel("Time $[s]$", 'Interpreter', 'latex');
ylabel("Longitudinal distance $[m]$", 'Interpreter', 'latex');
set(gca, "FontSize", 18, 'TickLabelInterpreter', 'latex');

%% Noise model - Gaussian
noiseModel = rl.option.GaussianActionNoise();
noiseModel.StandardDeviation = 0.3;
noiseModel.StandardDeviationDecayRate = 0.0001;
noiseModel.StandardDeviationMin = 0.05;

sampleCount = 40000;
standardDeviations = max(noiseModel.StandardDeviation .* (1 - noiseModel.StandardDeviationDecayRate) .^ (0:sampleCount-1), noiseModel.StandardDeviationMin);
x = noiseModel.Mean + randn(1, sampleCount) .* standardDeviations;
x = min(max(x, noiseModel.LowerLimit), noiseModel.UpperLimit);
plot(x)
xlabel("Samples", 'Interpreter', 'latex');
ylabel("Noise", 'Interpreter', 'latex');
grid on
set(gca, "FontSize", 18);

noiseGauss = x;

%% Noise model - Ohrstein-Uhlenbeck
noiseModel = rl.option.OrnsteinUhlenbeckActionNoise();
noiseModel.SampleTime = 0.05;
noiseModel.StandardDeviation = 0.3;
noiseModel.StandardDeviationDecayRate = 0.0001;
noiseModel.StandardDeviationMin = 0.05;
noiseModel.MeanAttractionConstant = 0.2;

sampleCount = 40000;
standardDeviations = max(noiseModel.StandardDeviation * (1 - noiseModel.StandardDeviationDecayRate) .^ (0:sampleCount-1), noiseModel.StandardDeviationMin);
randomNumbers = randn(1, sampleCount) .* standardDeviations .* sqrt(noiseModel.SampleTime);
x = zeros(1, sampleCount);
x(1) = noiseModel.InitialAction;
for k = 2:sampleCount
    x(k) = x(k-1) + noiseModel.MeanAttractionConstant .* (noiseModel.Mean - x(k-1)) .* noiseModel.SampleTime + randomNumbers(k);
end
plot(x)
xlabel("Samples", 'Interpreter', 'latex');
ylabel("Noise", 'Interpreter', 'latex');
grid on
set(gca, "FontSize", 18);

noiseOU = x;

%% Noises together
subplot(2, 1, 1);
plot(noiseGauss);
xlabel("Samples", 'Interpreter', 'latex');
ylabel("Gaussian", 'Interpreter', 'latex');
grid on
set(gca, "FontSize", 18);

subplot(2, 1, 2);
plot(noiseOU);
xlabel("Samples", 'Interpreter', 'latex');
ylabel("Ornstein-Uhlenbeck", 'Interpreter', 'latex');
grid on
set(gca, "FontSize", 18);

%% Trajectory plots
environment = CartPole.Environment();

%% Stable
environment.Mode = environment.ModeStable;
sampleCount = 160;
trajectory = environment.Trajectory.GenerateTrajectory(0, 3, 4);
initialObservation = [0;0;0;0];
[angles, massPositions, actions] = simulateEnvironment(environment, stableAgent, initialObservation, trajectory, sampleCount);

time = (0:sampleCount-1) * environment.Physics.Ts;

subplot(3, 1, 1);
plot(time, massPositions, "Color", "black", "LineWidth", 1.5);
hold on
plot(time, padArray(trajectory(1,:), sampleCount), "Color", "blue", "LineWidth", 1.5, "LineStyle", "--");
xlabel("Time [s]", "Interpreter", "latex");
ylabel({"Horizontal mass", "position [m]"}, "Interpreter", "latex");
legend({"Actual", "Reference"}, "Interpreter", "latex");
grid on
set(gca, "FontSize", 14);

subplot(3, 1, 2);
plot(time, angles/pi*180, "Color", "black", "LineWidth", 1.5);
xlabel("Time [s]", "Interpreter", "latex");
ylabel("Angle [$^{\circ}$]", "Interpreter", "latex");
grid on
set(gca, "FontSize", 14);

subplot(3, 1, 3);
stairs(time, actions, "Color", "black", "LineWidth", 1.5);
xlabel("Time [s]", "Interpreter", "latex");
ylabel({"Actuator", "force [N]"}, "Interpreter", "latex");
grid on
set(gca, "FontSize", 14);

%%
environment.Mode = environment.ModeStable;
sampleCount = 160;
time = (0:sampleCount-1) * environment.Physics.Ts;
trajectory = environment.Trajectory.GenerateTrajectory(0, 3, 4);

figure("Position", [596 386 1188 488]);
axes();
hold on;

physics = environment.Physics;
for i = 1:20
    environment.Physics = perturbPhysics(physics, 0.6);
    initialObservation = [0;0;(rand(1)-0.5)*2;0];
    [~, massPositions, ~] = simulateEnvironment(environment, stableAgent, initialObservation, trajectory, sampleCount);
    perturbedLine = plot(time, massPositions, "Color", "black");
end
environment.Physics = physics;

initialObservation = [0;0;0;0];
[~, massPositions, ~] = simulateEnvironment(environment, stableAgent, initialObservation, trajectory, sampleCount);

nominalLine = plot(time, massPositions, "Color", "red", "LineWidth", 2);
referenceLine = plot(time, padArray(trajectory(1,:), sampleCount), "Color", "blue", "LineWidth", 2, "LineStyle", "--");

xlabel("Time [s]", "Interpreter", "latex");
ylabel("Horizontal mass position [m]", "Interpreter", "latex");
legend([referenceLine, nominalLine, perturbedLine], {"Reference", "Nominal", "Perturbed"}, "Interpreter", "latex", "Location", "best");
grid on
set(gca, "FontSize", 14);

%%
environment.Mode = environment.ModeStable;
sampleCount = 120;
time = (0:sampleCount-1) * environment.Physics.Ts;

figure("Position", [596 386 1188 488]);
axes();
hold on;

physics = environment.Physics;
for i = 1:30
    environment.Physics = perturbPhysics(physics, 0.6);
    initialObservation = [0;0;pi+(rand(1)-0.5)*2;0];
    [angles, ~, ~] = simulateEnvironment(environment, stableAgent, initialObservation, [0;0], sampleCount);
    perturbedLine = plot(time, unwrap2(angles)/pi*180, "Color", "black");
end
environment.Physics = physics;

initialObservation = [0;0;pi;0];
[angles, ~, ~] = simulateEnvironment(environment, stableAgent, initialObservation, [0;0], sampleCount);


nominalLine = plot(time, unwrap2(angles)/pi*180, "Color", "red", "LineWidth", 2);

xlabel("Time [s]", "Interpreter", "latex");
ylabel("Angle [$^{\circ}$]", "Interpreter", "latex");
legend([nominalLine, perturbedLine], {"Nominal", "Perturbed"}, "Interpreter", "latex", "Location", "best");
grid on
set(gca, "FontSize", 14);

%% Unstable
environment.Mode = environment.ModeUnstable;
sampleCount = 160;
time = (0:sampleCount-1) * environment.Physics.Ts;
trajectory = environment.Trajectory.GenerateTrajectory(0, 2, 4);

figure("Position", [596 141 1188 733]);
axes1 = subplot(2, 1, 1);
hold on;
grid on
axes2 = subplot(2, 1, 2);
hold on;
grid on

physics = environment.Physics;
for i = 1:20
    environment.Physics = perturbPhysics(physics, 0.4);
    initialObservation = [0;0;pi+(rand(1)-0.5)*1;0];
    [angles, massPositions, ~] = simulateEnvironment(environment, unstableAgent, initialObservation, trajectory, sampleCount);
    perturbedLine1 = plot(axes1, time, massPositions, "Color", "black");
    perturbedLine2 = plot(axes2, time, unwrap2(angles)/pi*180+180, "Color", "black");
end
environment.Physics = physics;

initialObservation = [0;0;pi;0];
[angles, massPositions, ~] = simulateEnvironment(environment, unstableAgent, initialObservation, trajectory, sampleCount);

nominalLine1 = plot(axes1, time, massPositions, "Color", "red", "LineWidth", 2);
nominalLine2 = plot(axes2, time, unwrap2(angles)/pi*180+180, "Color", "red", "LineWidth", 2);
referenceLine = plot(axes1, time, padArray(trajectory(1,:), sampleCount), "Color", "blue", "LineWidth", 2, "LineStyle", "--");

xlabel(axes1, "Time [s]", "Interpreter", "latex");
ylabel(axes1, "Horizontal mass position [m]", "Interpreter", "latex");
legend([referenceLine, nominalLine1, perturbedLine1], {"Reference", "Nominal", "Perturbed"}, "Interpreter", "latex", "Location", "best");
set(axes1, "FontSize", 14);

xlabel(axes2, "Time [s]", "Interpreter", "latex");
ylabel(axes2, "Angle [$^{\circ}$]", "Interpreter", "latex");
legend(axes2, [nominalLine2, perturbedLine2], {"Nominal", "Perturbed"}, "Interpreter", "latex");
set(axes2, "FontSize", 14);

%%
environment.Mode = environment.ModeUnstable;
sampleCount = 160;
time = (0:sampleCount-1) * environment.Physics.Ts;

figure("Position", [596 386 1188 488]);
axes();
hold on;

physics = environment.Physics;
for i = 1:30
    environment.Physics = perturbPhysics(physics, 0.4);
    initialObservation = [0;0;(rand(1)-0.5)*2;0];
    [angles, ~, ~] = simulateEnvironment(environment, unstableAgent, initialObservation, [0;0], sampleCount);
    perturbedLine = plot(time, 180 + unwrap2(angles)/pi*180, "Color", "black");
end
environment.Physics = physics;

initialObservation = [0;0;0;0];
[angles, ~, ~] = simulateEnvironment(environment, unstableAgent, initialObservation, [0;0], sampleCount);

nominalLine = plot(time, 180 + unwrap2(angles)/pi*180, "Color", "red", "LineWidth", 2);

xlabel("Time [s]", "Interpreter", "latex");
ylabel("Angle [$^{\circ}$]", "Interpreter", "latex");
legend([nominalLine, perturbedLine], {"Nominal", "Perturbed"}, "Interpreter", "latex", "Location", "best");
grid on
set(gca, "FontSize", 14);


%%
function [angles, massPositions, actions] = simulateEnvironment(environment, agent, initialObservation, trajectory, sampleCount)
    observation = environment.reset(initialObservation, trajectory);
    angles = zeros(1, sampleCount);
    angles(1) = environment.State(3);
    massPositions = zeros(1, sampleCount);
    massPosition = environment.GetJointPosition();
    massPositions(1) = massPosition(1);
    actions = zeros(1, sampleCount);
    for i = 1:sampleCount-1
        actionCell = agent.getAction(observation);
        actions(i) = actionCell{1};
        observation = environment.step(actionCell{1});
        angles(i+1) = environment.State(3);
        massPosition = environment.GetJointPosition();
        massPositions(i+1) = massPosition(1);
    end
    actionCell = agent.getAction(observation);
    actions(end) = actionCell{1};
end

function physics = perturbPhysics(physics, maximalDeviation)
    randomNumbers = 1 + 2 * maximalDeviation * (rand(1, 5) - 0.5);
    physics.M0 = physics.M0 * randomNumbers(1);
    physics.b0 = physics.b0 * randomNumbers(2);
    physics.M1 = physics.M1 * randomNumbers(3);
    % physics.L1 = physics.L1 * randomNumbers(4);
    physics.b1 = physics.b1 * randomNumbers(5);
end
