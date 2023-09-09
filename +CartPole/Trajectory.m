classdef Trajectory < handle
    properties
        Parameter = struct( ...
            "Stable", struct(...
                "TimeLimits", [2, 3], ...
                "VelocityLimits", [0.5, 6], ...
                "StartingAngleDistributionWidth", 1, ...
                "StandstillProbability", 0, ...
                "UnstableStartingPositionProbability", 0.0), ...
            "Unstable", struct( ...
                "StableStartingAngleDistributionWidth", 0.5, ...
                "UnstableStartingAngleDistributionWidth", 0.5, ...
                "StandstillProbability", 1, ...
                "StableStartingPositionProbability", 0, ...
                "TimeLimits", [3, 5], ...
                "VelocityLimits", [0.2, 1]))
    end

    properties (Access = protected)
        Environment_
    end

    properties (Constant)
        TransitionCurve = makima(...
            [-3,-2,-1,0,1,2,3], ...
            [0,0,0,0,1,1,1])
        MaximalDerivative = 1.5
    end
    
    methods
        function this = Trajectory(environment)
            this.Environment_ = environment;
        end

        function [initialState, trajectory] = GenerateStable(this)
            randomNumbers = rand(3, 1);
            
            probabilityThresholds = [cumsum( [ ...
                this.Parameter.Stable.UnstableStartingPositionProbability, ...
                this.Parameter.Stable.StandstillProbability ...
                ]) 1];
            switch find(randomNumbers(1) <= probabilityThresholds, 1)
                case 1 % Stabilize starting from unstable position
                    workspaceWidth = this.Environment_.Workspace(3) - this.Environment_.Workspace(1);
                    
                    % Limit starting distance to the middle 80% of horizontal workspace
                    startingDistance = this.Environment_.Workspace(1) + 0.1 * workspaceWidth + ...
                        workspaceWidth * 0.8 * randomNumbers(2);
                    trajectory = [ ...
                        startingDistance; ... x1_ref
                        0; ... x1d_ref
                        ];
    
                    startingAngle = pi + this.Parameter.Stable.StartingAngleDistributionWidth * (randomNumbers(3) - 0.5);
                    startingAngle = this.Environment_.CentralizeAngle(startingAngle, 0);
    
                    initialState = [ ...
                        startingDistance; ... x0
                        0; ... x0d
                        startingAngle; ... phi1
                        0; ... phi1d
                        ];
                case 2 % Standstill
                    workspaceWidth = this.Environment_.Workspace(3) - this.Environment_.Workspace(1);
                    
                    % Limit starting distance to the middle 80% of horizontal workspace
                    startingDistance = this.Environment_.Workspace(1) + 0.1 * workspaceWidth + ...
                        workspaceWidth * 0.8 * randomNumbers(2);
                    trajectory = [ ...
                        startingDistance; ... x1_ref
                        0; ... x1d_ref
                        ];

                    startingAngle = this.Parameter.Stable.StartingAngleDistributionWidth * (randomNumbers(3) - 0.5);
                    startingAngle = this.Environment_.CentralizeAngle(startingAngle, 0);
    
                    initialState = [ ...
                        startingDistance; ... x0
                        0; ... x0d
                        startingAngle; ... phi1
                        0; ... phi1d
                        ];
                otherwise % Follow smooth trajectory starting from stable position
                    trajectory = this.GenerateSmoothTransition_( ...
                        this.Parameter.Stable.TimeLimits, ...
                        this.Parameter.Stable.VelocityLimits);
                    
                    startingDistance = trajectory(1, 1);
                    startingAngle = this.Parameter.Stable.StartingAngleDistributionWidth * (randomNumbers(2) - 0.5);
                    
                    initialState = [ ...
                        startingDistance; ... x0
                        0; ... x0d
                        startingAngle; ... phi1
                        0; ... phi1d
                        ];
            end
        end

        function [initialState, trajectory] = GenerateUnstable(this)
            randomNumbers = rand(3, 1);

            probabilityThresholds = [cumsum( [ ...
                this.Parameter.Unstable.StableStartingPositionProbability, ...
                this.Parameter.Unstable.StandstillProbability ...
                ]) 1];
            switch find(randomNumbers(1) <= probabilityThresholds, 1)
                case 1 % Swing pendulum upwards from stable position
                    startingAngle = this.Parameter.Unstable.StableStartingAngleDistributionWidth * (randomNumbers(2) - 0.5);

                    % Limit starting distance to the middle 50% of horizontal workspace
                    workspaceWidth = this.Environment_.Workspace(3) - this.Environment_.Workspace(1);
                    startingDistance = this.Environment_.Workspace(1) + 0.25 * workspaceWidth + ...
                        workspaceWidth * 0.5 * randomNumbers(3);
    
                    trajectory = [ ...
                        startingDistance; ... x1_ref
                        0; ... x1d_ref
                        ];
                case 2 % Standstill starting from unstable position
                    startingAngle = pi + this.Parameter.Unstable.UnstableStartingAngleDistributionWidth * (randomNumbers(2) - 0.5);
                    startingAngle = this.Environment_.CentralizeAngle(startingAngle, 0);
                    
                    % Limit starting distance to the middle 50% of horizontal workspace
                    workspaceWidth = this.Environment_.Workspace(3) - this.Environment_.Workspace(1);
                    startingDistance = this.Environment_.Workspace(1) + 0.25 * workspaceWidth + ...
                        workspaceWidth * 0.5 * randomNumbers(3);

                    trajectory = [ ...
                        startingDistance; ... x1_ref
                        0; ... x1d_ref
                        ];
                otherwise % Follow smooth trajectory starting from unstable position
                    trajectory = this.GenerateSmoothTransition_( ...
                        this.Parameter.Unstable.TimeLimits, ...
                        this.Parameter.Unstable.VelocityLimits);
                    startingDistance = trajectory(1, 1);
    
                    startingAngle = pi + this.Parameter.Unstable.UnstableStartingAngleDistributionWidth * (randomNumbers(3) - 0.5);
                    startingAngle = this.Environment_.CentralizeAngle(startingAngle, 0);
            end
    
            initialState = [ ...
                startingDistance; ... x0
                0; ... x0d
                startingAngle; ... phi1
                0; ... phi1d
                ];
        end
    end

    methods (Access = protected)
        function trajectory = GenerateSmoothTransition_(this, timeLimits, velocityLimits)
            workspaceWidth = this.Environment_.Workspace(3) - this.Environment_.Workspace(1);

            distanceLimits = [ ...
                timeLimits(2) * velocityLimits(1), ...
                timeLimits(1) * velocityLimits(2)] ...
                / this.MaximalDerivative;
            if distanceLimits(2) >= workspaceWidth * 0.8
                error("Inputs resulted in maximal distance (%d) larger than 80% of workspace width (%d)!", ...
                    distanceLimits(2), workspaceWidth * 0.8);
            end

            randomNumbers = rand(4, 1);

            duration = timeLimits(1) + diff(timeLimits) * randomNumbers(1);
            distance = distanceLimits(1) + diff(distanceLimits) * randomNumbers(2);
            
            % Limit starting distance to the middle 80% of horizontal workspace
            startingDistance = this.Environment_.Workspace(1) + 0.1 * workspaceWidth + ...
                (workspaceWidth * 0.8 - distance) * randomNumbers(3); 
            
            % Flip starting and ending positions wwith 50% probability
            positionSwitch = randomNumbers(4) > 0.5;
            startingDistance = startingDistance + distance * positionSwitch;
            if positionSwitch
                distance = -distance;
            end

            transitionSampleCount = floor(duration / this.Environment_.Physics.Ts) + 1;
            transitionTimeSamples = linspace(0, 1, transitionSampleCount);
            trajectory = zeros(2, transitionSampleCount);
            trajectory(1,:) = startingDistance + distance * ...
                ppval(this.TransitionCurve, transitionTimeSamples); % x1_ref
            trajectory(2,:) = [0 (-trajectory(1,1:end-2)+trajectory(1,3:end))/2/this.Environment_.Physics.Ts 0]; % x1d_ref
        end
    end
end

% environment = CartPole.Environment();
% 
% %%
% [initialObservation, trajectory] = environment.Trajectory.GenerateStable();
% 
% Ts = environment.Physics.Ts;
% plot(0:Ts:(length(trajectory) - 1) * Ts, trajectory(1,:));
% hold on
% disp(max(abs(trajectory(2,:))))
% 
% %%
% plot(trajectory(2,:))
% hold on
% plot([0 (-trajectory(1,1:end-2) + trajectory(1,3:end))/2 0]/Ts)