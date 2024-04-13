function Offspring = DRSolutionGeneration(Problem, Population, IndexDif, P, svm_mdl, R_max, i)
% Solution-generation in DR-MCEA/D

%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Yuma Horaguchi

    for r = 1 : R_max
        % Generate candidate solution
        candidate      = OperatorDE(Problem, Population(i).dec, Population(P(1)).dec, Population(P(2)).dec);

        % Shuffle the parents
        rnd            = randperm(length(P));
        P              = P(rnd);

        % Input the candidate solution to SVM
        uniform_cand   = (candidate - Problem.lower) ./ (Problem.upper - Problem.lower);
        [class, score] = svm_mdl.predict(uniform_cand(:, IndexDif));
        score          = score(2);
  
        if class == 1
            % If predicted label of the candidate solution is positive class
            % Return the candidate solution and terminate the process
            Offspring = candidate;
            return
        else
            % If predicted label of the candidate solution is negative class
            % Choose the candidate solution having the best decision score function value
            if r == 1
                score_max = score;
                Offspring = candidate;
            elseif score_max < score
                score_max = score;
                Offspring = candidate;
            end
        end
    end
end