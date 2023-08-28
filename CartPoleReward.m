classdef CartPoleReward < handle
    properties
        Parameter = struct( ...
            "Stable", struct(...
                "Gainx1", 1, ...
                "Gainy1", 1, ...
                "Gainx0d", 0.2, ...
                "Gainx1d", 0.2, ...
                "GainF", 0.1, ...
                "PenaltyOutOfBoundary", 0, ...
                "RewardWithinBoundary", 10), ...
            "Unstable", struct(...
                "Gainx1", 1, ...
                "Gainy1", 1, ...
                "Gainx0d", 0.2, ...
                "Gainx1d", 0.2, ...
                "GainF", 0.1, ...
                "PenaltyOutOfBoundary", 0, ...
                "RewardWithinBoundary", 20, ...
                "UnstableAngleRegionHalfWidth", pi/4))

        Function = @(action, state, trajectory, observation) 0
    end

    properties (Access = protected)
        Environment_
    end

    methods
        function this = CartPoleReward(environment)
            this.Environment_ = environment;
        end
        
        function [reward, isDone] = CalculateStable(this, action, state, trajectory, observation)
            % x0_ = state(1);
            % x0d_ = state(2);
            % phi1_ = state(3); % already centered around 0
            % phi1d_ = state(4);
            
            x0e_ = observation(1);
            x0de_ = observation(2);
            x1e_ = observation(3);
            x1de_ = observation(4);
            y1_ = observation(5);
            % y1d_ = observation(6);
            
            y1e_ = - this.Environment_.Physics.L1 - y1_; % Relative to lower stable position

            reward = this.Parameter.Stable.RewardWithinBoundary - ( ...
                this.Parameter.Stable.Gainx1 * x1e_^2 + ...
                this.Parameter.Stable.Gainy1 * y1e_^2 + ...
                this.Parameter.Stable.Gainx0d * x0de_^2 + ...
                this.Parameter.Stable.Gainx1d * x1de_^2 + ...
                this.Parameter.Stable.GainF * action^2);

            % Out of bounds
            isDone = abs(x0e_) >= (this.Environment_.Workspace(3) / 2); % Need to contain horizontal position
            if isDone
                reward = reward - this.Parameter.Stable.PenaltyOutOfBoundary;
            end
        end

        function [reward, isDone] = CalculateUnstable(this, action, state, trajectory, observation)
            % x0_ = state(1);
            % x0d_ = state(2);
            phi1_ = state(3); % already centered around 0
            % phi1d_ = state(4);
            
            x0e_ = observation(1);
            x0de_ = observation(2);
            x1e_ = observation(3);
            x1de_ = observation(4);
            y1_ = observation(5);
            % y1d_ = observation(6);
            
            y1e_ = this.Environment_.Physics.L1 - y1_; % Relative to upper stable position

            reward = this.Parameter.Unstable.RewardWithinBoundary - ( ...
                this.Parameter.Unstable.Gainx1 * x1e_^2 + ...
                this.Parameter.Unstable.Gainy1 * y1e_^2 + ...
                this.Parameter.Unstable.Gainx0d * x0de_^2 + ...
                this.Parameter.Unstable.Gainx1d * x1de_^2 + ...
                this.Parameter.Unstable.GainF * action^2);
    
            % Check unstable region entry
            isWithinUnstableRegion = false;
            if abs(this.Environment_.CentralizeAngle(phi1_, pi)) < this.Parameter.Unstable.UnstableAngleRegionHalfWidth
                isWithinUnstableRegion = true;
                this.Environment_.HasAlreadyEnteredUnstableRegion = true;
            end

            % Left unstable region
            isDone = ~isWithinUnstableRegion && this.Environment_.HasAlreadyEnteredUnstableRegion;
            if isDone
                reward = reward - this.Parameter.Stable.PenaltyOutOfBoundary;
                return
            end

            % Out of bounds
            isDone = abs(x0e_) >= (this.Environment_.Workspace(3)); % Need to contain horizontal position
            if isDone
                reward = reward - this.Parameter.Stable.PenaltyOutOfBoundary;
            end
        end
    end
end