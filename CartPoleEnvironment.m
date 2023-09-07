classdef CartPoleEnvironment < rl.env.MATLABEnvironment
    properties
        % Stable: 1
        % Unstable: 2
        Mode = 1

        Physics = struct( ...
            'M0', 0.1, ... [kg]
            'b0', 0.1, ... [Ns/m]
            'g', 9.81, ... [m/s2]
            'M1', 0.1, ... [kg]
            'L1', 2, ... [m]
            'b1', 0.01, ... [Nms]
            'Fmax', 5, ... [N]
            'Ts', 0.05) % [s]
        
        Workspace = [-10, -2, 10, 2] % [lowerLeftCornerXY, upperRightCornerXY]

        TrajectoryCollection = []
        Reward = []
    end
    
    properties(Constant)
        ObservationDimension = 6
        StateDimension = 4
        ActionDimension = 1
    end

    properties (Access = protected)
        State_ = zeros(CartPoleEnvironment.StateDimension, 1) % [x0, x0d, phi1, phi1d]'
        Action_ = 0
        Trajectory_ = []

        SimulationTime_ = 0
        SimulationStepCount_ = 1
        Figure_ = []
        Axes_ = []
        Timer_ = []

        PositionReference_ = 0
        LastPositionReferenceMean_ = 0
        PositionReferenceBuffer_ = []
        Agent_ = []
        AgentIndex_ = 0
    end

    properties (Access = public)
        HasAlreadyEnteredUnstableRegion = false
    end

    methods
        function this = CartPoleEnvironment()
            ObservationInfo = rlNumericSpec([CartPoleEnvironment.ObservationDimension 1]);
            ObservationInfo.Name = 'CartPoleEnvironment observation';
            ObservationInfo.Description = 'x1_ref - x1, x1d_ref - x1d, y1, y1d';
            
            ActionInfo = rlNumericSpec([CartPoleEnvironment.ActionDimension, 1]);
            ActionInfo.Name = 'CartPoleEnvironment action';
            ActionInfo.Description = 'F';

            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
            
            this.ActionInfo.LowerLimit = - this.Physics.Fmax;
            this.ActionInfo.UpperLimit = + this.Physics.Fmax;

            this.TrajectoryCollection = CartPoleTrajectoryCollection(this);
            this.Reward = CartPoleReward(this);
        end

        function jointPosition = GetJointPosition(this)
            x0_ = this.State_(1);
            phi1_ = this.State_(3);

            L1 = this.Physics.L1;
            
            jointPosition = [ ...
                x0_ + L1*sin(phi1_); ... x1
                - L1*cos(phi1_) ... y1
                ];
        end
        function jointVelocity = GetJointVelocity(this)
            x0d_ = this.State_(2);
            phi1_ = this.State_(3);
            phi1d_ = this.State_(4);

            L1 = this.Physics.L1;

            jointVelocity = [ ...
                x0d_ + L1 * phi1d_ * cos(phi1_); ... x1d
                L1 * phi1d_ * sin(phi1_) ... y1d
                ];
        end
        
        function InteractiveForce(this)
            if ~isempty(this.Figure_) && isvalid(this.Figure_)
                return
            end
            
            this.State_ = zeros(4, 1);
            this.Action_ = 0;
            this.Draw();
            
            function StepSimulationDraw_(this, tend)
                this.StepSimulation_(tend);
                this.Draw();
            end
            this.Timer_ = timer("ExecutionMode", "fixedRate", ...
                "Period", 0.1, ... 10 Hz
                "TimerFcn", @(src, ~) StepSimulationDraw_(this, src.Period), ...
                "ErrorFcn", @(src, ~) delete(src));

            function SetForce_(this, F)
                this.Action_ = F;
            end
            manipulate(this.Figure_, @(F) SetForce_(this, F), {'F', -this.Physics.Fmax, this.Physics.Fmax, 0});

            start(this.Timer_);
        end
        function InteractiveAgent(this, agent)
            if ~isempty(this.Figure_) && isvalid(this.Figure_)
                return
            end

            this.Agent_ = agent;
            this.AgentIndex_ = 0;
            
            this.SimulationTime_ = 0;
            this.State_ = zeros(4, 1);
            this.State_(3) = -0.1;
            this.Action_ = 0;
            this.Draw();
            
            x0 = this.State_(1);
            this.PositionReferenceBuffer_ = CircularBuffer(ones(1, 10) * x0);
            this.LastPositionReferenceMean_ = x0;
            
            function StepSimulationDraw_(this)
                while this.SimulationTime_ <= this.Timer_.Period
                    this.SimulationTime_ = this.SimulationTime_ + this.Physics.Ts;

                    % Position reference
                    this.PositionReferenceBuffer_.Push(this.PositionReference_);
                    positionReferenceMean = mean(this.PositionReferenceBuffer_.IndexedData);
                    positionReferenceVelocity = (positionReferenceMean - this.LastPositionReferenceMean_) / this.Physics.Ts;
                    this.LastPositionReferenceMean_ = positionReferenceMean;
                    
                    % Observation
                    observation = this.GetObservation([positionReferenceMean; positionReferenceVelocity]);

                    % Action
                    actionCell = this.Agent_.getAction(observation);
                    this.Action_ = actionCell{1};
                    this.StepSimulation_(this.Physics.Ts);
                end

                this.SimulationTime_ = this.SimulationTime_ - this.Timer_.Period;
                this.Draw();
            end
            this.Timer_ = timer("ExecutionMode", "fixedRate", ...
                "Period", 0.1, ... 10 Hz
                "TimerFcn", @(src, ~) StepSimulationDraw_(this), ...
                "ErrorFcn", @(src, ~) delete(src));

            function SetPositionReference_(this, x1_ref)
                this.PositionReference_ = x1_ref;
            end
            manipulate(this.Figure_, @(x1_ref) SetPositionReference_(this, x1_ref), {'x1_ref', this.Workspace(1), this.Workspace(3), x0});

            start(this.Timer_);
        end

        function InteractiveDoubleAgent(this, stableAgent, unstableAgent)
            if ~isempty(this.Figure_) && isvalid(this.Figure_)
                return
            end

            this.Agent_ = {stableAgent, unstableAgent};
            this.AgentIndex_ = 1;
            
            this.SimulationTime_ = 0;
            this.State_ = zeros(4, 1);
            this.State_(3) = 0;
            this.Action_ = 0;
            this.Draw();
            
            x0 = this.State_(1);
            this.PositionReferenceBuffer_ = CircularBuffer(ones(1, 10) * x0);
            this.LastPositionReferenceMean_ = x0;
            
            function StepSimulationDraw_(this)
                while this.SimulationTime_ <= this.Timer_.Period
                    this.SimulationTime_ = this.SimulationTime_ + this.Physics.Ts;

                    % Position reference
                    this.PositionReferenceBuffer_.Push(this.PositionReference_);
                    positionReferenceMean = mean(this.PositionReferenceBuffer_.IndexedData);
                    positionReferenceVelocity = (positionReferenceMean - this.LastPositionReferenceMean_) / this.Physics.Ts;
                    this.LastPositionReferenceMean_ = positionReferenceMean;
                    
                    % Observation
                    observation = this.GetObservation([positionReferenceMean; positionReferenceVelocity]);

                    % Action
                    actionCell = this.Agent_{this.AgentIndex_}.getAction(observation);
                    this.Action_ = actionCell{1};
                    this.StepSimulation_(this.Physics.Ts);
                end

                this.SimulationTime_ = this.SimulationTime_ - this.Timer_.Period;
                this.Draw();
            end
            this.Timer_ = timer("ExecutionMode", "fixedRate", ...
                "Period", 0.1, ... 10 Hz
                "TimerFcn", @(src, ~) StepSimulationDraw_(this), ...
                "ErrorFcn", @(src, ~) delete(src));

            function SetPositionReference_(this, x1_ref, y1_ref)
                this.PositionReference_ = x1_ref;

                if y1_ref < -0.9
                    this.AgentIndex_ = 1; % stable agent
                elseif y1_ref > 0.9
                    this.AgentIndex_ = 2; % unstable agent
                end
            end
            manipulate(this.Figure_, @(x1_ref, y1_ref) SetPositionReference_(this, x1_ref, y1_ref), ...
                {'x1_ref', this.Workspace(1), this.Workspace(3), x0}, ...
                {'y1_ref', -1, 1, -1});

            start(this.Timer_);
        end

        function Draw(this)
            if isempty(this.Figure_) || ~isvalid(this.Figure_)
                this.CreateFigure_();
            end

            this.UpdateFigure_();
        end

        function observation = GetObservation(this, trajectory)
            x0_ = this.State_(1);
            x0d_ = this.State_(2);
            jointPosition = this.GetJointPosition();
            jointVelocity = this.GetJointVelocity();

            observation = [ ...
                trajectory(1) - x0_; ... x1_ref - x0 (horizontal relative cart position)
                trajectory(2) - x0d_; ... x1d_ref - x0d (horizontal relative cart velocity)
                trajectory(1) - jointPosition(1); ... x1_ref - x1 (horizontal position tracking error)
                trajectory(2) - jointVelocity(1); ... x1d_ref - x1d (horizontal velocity tracking error)
                jointPosition(2); ... y1
                jointVelocity(2); ... y1d
                ];
        end
    end
    
    
    methods % rl.env.MATLABEnvironment
        function [observation, reward, isDone, loggedSignals] = step(this, action)
            loggedSignals = [];

            this.Action_ = action;
            this.StepSimulation_(this.Physics.Ts);
            this.SimulationTime_ = this.SimulationTime_ + this.Physics.Ts;
            this.SimulationStepCount_ = this.SimulationStepCount_ + 1;
            
            trajectory = this.Trajectory_(:, min(this.SimulationStepCount_, size(this.Trajectory_, 2)));
            observation = this.GetObservation(trajectory);
            
            [reward, isDone] = this.Reward.Function(...
                this.Action_, ...
                this.State_, ...
                trajectory, ...
                observation ...
                );
        end
        
        function initialObservation = reset(this)
            switch this.Mode
                case 1
                    [initialState, this.Trajectory_] = this.TrajectoryCollection.GenerateStable();
                    this.Reward.Function = @(action, state, trajectory, observation) this.Reward.CalculateStable(action, state, trajectory, observation);
                case 2
                    [initialState, this.Trajectory_] = this.TrajectoryCollection.GenerateUnstable();
                    this.Reward.Function = @(action, state, trajectory, observation) this.Reward.CalculateUnstable(action, state, trajectory, observation);
                    this.HasAlreadyEnteredUnstableRegion = false; % needed for episode termination check
                otherwise
                    error("Mode %d not supported.", this.Mode);
            end

            this.SimulationTime_ = 0;
            this.SimulationStepCount_ = 1;
            this.Action_ = 0;
            this.State_ = initialState;
            initialObservation = this.GetObservation(this.Trajectory_(:,1));
        end
    end

    methods (Access = protected)
        function dydt = odefun_(this, y, F)
            % x0_ = y(1);
            x0d_ = y(2);
            phi1_ = y(3);
            phi1d_ = y(4);

            sin_phi1_ = sin(phi1_);
            cos_phi1_ = cos(phi1_);

            M0 = this.Physics.M0;
            b0 = this.Physics.b0;
            M1 = this.Physics.M1;
            g = this.Physics.g;
            L1 = this.Physics.L1;
            b1 = this.Physics.b1;

            mass_determinant = L1*(- M1*cos_phi1_.^2 + M0 + M1);
            
            dydt = [ ...
                x0d_; ...
                (L1*F - L1*b0.*x0d_ + b1.*cos_phi1_.*phi1d_ + L1^2*M1.*(phi1d_.^2).*sin_phi1_ + L1*M1.*cos_phi1_*g.*sin_phi1_)./mass_determinant; ...
                phi1d_; ...
                -((M0 + M1)*b1.*phi1d_ + L1*M1.*cos_phi1_.*F + L1*M1^2*g.*sin_phi1_ + L1*M0*M1*g.*sin_phi1_ - L1*M1*b0.*cos_phi1_.*x0d_ + L1^2*M1^2.*cos_phi1_.*(phi1d_.^2).*sin_phi1_)./(L1*M1.*mass_determinant)];
        end
        function StepSimulation_(this, tend)
            [~, y_] = ode45(@(~, y) this.odefun_(y, this.Action_), ...
                [0, tend], this.State_);
            this.State_ = y_(end, :)';

            % Bound angle
            this.State_(3) = this.CentralizeAngle(this.State_(3), 0);
        end

        function CreateFigure_(this)
            this.Figure_ = figure('Position', [325         354        1276         410], ...
                'CloseRequestFcn', @(~, ~) this.onFigureClose_, ...
                'Name', 'Cart pole control', 'NumberTitle', 'off', ...
                'Color', [1, 1, 1], ...
                'MenuBar', 'none', 'ToolBar', 'none');
            this.Axes_ = axes('Parent', this.Figure_, ...
                'Position', [0, 0, 1, 1], ...
                'XColor', [1, 1, 1], 'YColor', [1, 1, 1], ...
                'XTick', [], 'YTick', [], ...
                'XLimMode', 'manual', 'YLimMode', 'manual', ...
                'XLim', 1.2*this.Workspace([1 3]), 'YLim', 1.2*this.Workspace([2 4]), ...
                'DataAspectRatioMode', 'manual', 'DataAspectRatio', [1, 1, 1]); % 'XLimMode', 'manual', 'YLimMode', 'manual',
            % axis(ax, 'equal');
            box off

            L1 = this.Physics.L1;

            this.Axes_.UserData = struct();
            this.Axes_.UserData.Cart = rectangle('Parent', this.Axes_, ...
                'Position', [0, 0, L1*0.2*1.6180, L1*0.2], ...
                'LineWidth', 1, ...
                'Curvature', [0.1, 0.1], ...
                'EdgeColor', [0.2, 0.2, 0.2], ...
                'FaceColor', [0.9, 0.9, 0.9]);
            this.Axes_.UserData.Pole = line('Parent', this.Axes_, ...
                'XData', [0, 0], 'YData', [0, 0], ...
                'Color', [0.5, 0.5, 0.5], ...
                'LineWidth', 2);
            radius = L1*0.1;
            this.Axes_.UserData.Mass1 = rectangle('Parent', this.Axes_, ...
                'Position', [0, 0, 2*radius, 2*radius], ...
                'Curvature', [1, 1], ...
                'EdgeColor', [0.2, 0.2, 0.2], ...
                'FaceColor', [0.9, 0.9, 0.9]);
            this.Axes_.UserData.Trajectory = animatedline(this.Axes_, ...
                "LineStyle", ":", ...
                "MaximumNumPoints", 100);
            if this.AgentIndex_ > 0
                this.Axes_.UserData.Target = rectangle('Parent', this.Axes_, ...
                    'Position', [0, 0, 2*radius, 2*radius]*1.1, ...
                    'Curvature', [1, 1], ...
                    'EdgeColor', 'b', ...
                    'LineStyle', '--', ...
                    'LineWidth', 1);
            else
                this.Axes_.UserData.Target = xline(0, 'Parent', this.Axes_, ...
                    'LineStyle', ':', ...
                    'Color', 'b');
            end
            yline(this.Axes_, -this.Axes_.UserData.Cart.Position(4)/2,'--');
        end
        function onFigureClose_(this)
            if ~isempty(this.Timer_) && isvalid(this.Timer_)
                stop(this.Timer_);
                delete(this.Timer_);
            end
            delete(this.Figure_);
        end
        function UpdateFigure_(this)
            x0 = this.State_(1);

            jointPosition = this.GetJointPosition();
            x1 = jointPosition(1);
            y1 = jointPosition(2);

            MoveRectangle(this.Axes_.UserData.Cart, [x0, 0]);
            set(this.Axes_.UserData.Pole, ...
                'XData', [x0, x1], ...
                'YData', [0, y1]);
            MoveRectangle(this.Axes_.UserData.Mass1, [x1, y1]);
            addpoints(this.Axes_.UserData.Trajectory, x1, y1);

            switch this.AgentIndex_
                case 0
                    this.Axes_.UserData.Target.Value = this.LastPositionReferenceMean_;
                case 1 % Stable
                    MoveRectangle(this.Axes_.UserData.Target, [this.LastPositionReferenceMean_, -this.Physics.L1]);
                case 2 % Unstable
                    MoveRectangle(this.Axes_.UserData.Target, [this.LastPositionReferenceMean_, this.Physics.L1]);
            end
        end
    end

    methods(Static)
        function angle = CentralizeAngle(angle, center)
            angle = mod(angle - center - pi, 2*pi) - pi;
        end

        function [solution, latex_] = GetEquationsOfMotion()
            latex_ = struct();

            syms g M0 b0 M1 L1 b1 x0(t) phi1(t) F(t)

            p0 = [x0(t);
                  0];
            p1 = [p0(1) + L1*sin(phi1(t));
                  p0(2) - L1*cos(phi1(t))];
            
            v0 = diff(p0, 1);
            v1 = diff(p1, 1);
            
            real_dot = @(u, v) u(1)*v(1) + u(2)*v(2);
            
            T = simplify(M0*real_dot(v0, v0)/2 + M1*real_dot(v1, v1)/2);
            D = b0*diff(x0(t), t)^2/2 + b1*diff(phi1(t), t)^2/2;
            U = M1*g*p1(2);
            
            eulerLagrange = @(v, Q) simplify(...
                diff(diff(T, diff(v, t)), t) ...
                - diff(T, v) ...
                + diff(D, diff(v, t)) ...
                + diff(U, v) == Q);
            
            syms x0dd_ phi1dd_
            eq_x0 = eulerLagrange(x0, F(t));
            eq_x0 = subs(eq_x0, diff(x0, t, t), x0dd_);
            eq_x0 = subs(eq_x0, diff(phi1, t, t), phi1dd_);
            
            eq_phi1 = eulerLagrange(phi1, 0);
            eq_phi1 = subs(eq_phi1, diff(x0, t, t), x0dd_);
            eq_phi1 = subs(eq_phi1, diff(phi1, t, t), phi1dd_);
            
            solution = solve([eq_x0, eq_phi1], [x0dd_, phi1dd_]);
            solution.x0dd_ = simplify(solution.x0dd_);
            solution.phi1dd_ = simplify(solution.phi1dd_);
            
            latex_.x0dd_ = latex(diff(x0(t), t, t) == solution.x0dd_);
            latex_.phi1dd_ = latex(diff(phi1(t), t, t) == solution.phi1dd_);
            
            syms x0_ x0d_ phi1_ phi1d_ sin_phi1_ cos_phi1_
            solution.x0dd_ = subs(solution.x0dd_, diff(x0(t), t), x0d_);
            solution.x0dd_ = subs(solution.x0dd_, x0(t), x0_);
            solution.x0dd_ = subs(solution.x0dd_, diff(phi1(t), t), phi1d_);
            solution.x0dd_ = subs(solution.x0dd_, phi1(t), phi1_);
            solution.x0dd_ = subs(solution.x0dd_, sin(phi1_), sin_phi1_);
            solution.x0dd_ = subs(solution.x0dd_, cos(phi1_), cos_phi1_);

            solution.phi1dd_ = subs(solution.phi1dd_, diff(x0(t), t), x0d_);
            solution.phi1dd_ = subs(solution.phi1dd_, x0(t), x0_);
            solution.phi1dd_ = subs(solution.phi1dd_, diff(phi1(t), t), phi1d_);
            solution.phi1dd_ = subs(solution.phi1dd_, phi1(t), phi1_);
            solution.phi1dd_ = subs(solution.phi1dd_, sin(phi1_), sin_phi1_);
            solution.phi1dd_ = subs(solution.phi1dd_, cos(phi1_), cos_phi1_);
        end
    end
end

