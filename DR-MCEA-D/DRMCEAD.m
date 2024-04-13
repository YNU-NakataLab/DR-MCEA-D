classdef DRMCEAD < ALGORITHM
% <multi/many> <real/integer> <expensive>
% Dimensionality Reduction-based MCEA/D 
% delta  --- 0.9 --- The probability of choosing parents locally
% nr     ---   2 --- Maximum number of solutions replaced by each offspring
% Rmax   ---  10 --- Maximum repeat time of offspring generation
% C      --- 1.0 --- The parameter for SVM
% gamma  --- 1.0 --- The parameter for SVM
% beta   --- 0.5 --- The reduction rate
% rho    --- 0.1 --- The rate of the superior solutions and the inferior solutions

%------------------------------- Reference --------------------------------
% Y. Horaguchi and M. Nakata, High-Dimensional Expensive Optimization by 
% Classification-based Multiobjective Evolutionary Algorithm with 
% Dimensionality Reduction, 2023 62nd Annual Conferenc of the Society of 
% Instrument and Control Engnieers (SICE), 2023, 1535-1542.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Yuma Horaguchi

    methods
        function main(Algorithm, Problem)
            %% Parameter setting
            [delta, nr, R_max, C, gamma, beta, rho] = Algorithm.ParameterSet(0.9, 2, 10, 1.0, 1.0, 0.5, 0.1);
            
            %% Generate the weight vectors
            [W, Problem.N] = UniformPoint(Problem.N, Problem.M);
        
            %% Detect the neighbours of each solution
            T      = ceil(Problem.N / 10);
            B      = pdist2(W, W);
            [~, B] = sort(B, 2);
            B      = B(:, 1 : T);
        
            %% Initialize population
            PopDec     = UniformPoint(Problem.N, Problem.D, 'Latin');
            Population = Problem.Evaluation(repmat(Problem.upper - Problem.lower, Problem.N, 1) .* PopDec + repmat(Problem.lower, Problem.N, 1));
            Arc        = Population;
            Z          = min(Population.objs, [], 1);
            sigma      = sqrt(1 / (2 * gamma));
            
            %% Optimization
            while Algorithm.NotTerminated(Arc)
                % For each sub-problem
                for i = 1 : Problem.N
                    %% Dimension Reduction
                    IndexDif = DimensionalityReduction(Problem, Arc, beta, rho, W, Z, i);

                    %% Build a SVM classifier
                    C_i   = [];
                    label = -1 * ones(length(Arc), 1);
                    for k = 1 : length(B(i, :))
                        g_A        = max(abs(Arc.objs - repmat(Z, length(Arc), 1)) .* W(B(i, k), :), [], 2);
                        [~, rankA] = sort(g_A);
                        for j = 1 : length(Arc)
                            if ~ismember(rankA(j), C_i)
                                C_i = [C_i, rankA(j)];
                                label(rankA(j)) = +1;
                                break
                            end
                        end
                    end
                    uniform_ADec = (Arc.decs - Problem.lower) ./ (Problem.upper - Problem.lower);
                    svm_mdl      = fitcsvm(uniform_ADec(:, IndexDif), label, 'BoxConstraint', C, 'KernelScale', sigma, 'KernelFunction', 'rbf');
                    
                    %% Choose the parents
                    if rand < delta
                        P = B(i, randperm(end));
                    else
                        P = randperm(Problem.N);
                    end
        
                    %% Solution-generation
                    Offspring = DRSolutionGeneration(Problem, Population, IndexDif, P, svm_mdl, R_max, i);
        
                    %% Evaluate offspring
                    Offspring = Problem.Evaluation(Offspring);
        
                    %% Update the reference point
                    Z = min(Z, Offspring.obj);
        
                    %% Update population and archive
                    g_old = max(abs(Population(P).objs - repmat(Z, length(P), 1)) .* W(P, :), [], 2);
                    g_new = max(repmat(abs(Offspring.obj - Z), length(P), 1) .* W(P, :), [], 2);
                    Population(P(find(g_old >= g_new, nr))) = Offspring;
                    Arc   = [Arc, Offspring];
                    
                    %% Check termination criteria
                    Algorithm.NotTerminated(Arc);
                end
            end
        end
    end
end