function IndexDif = DimensionalityReduction(Problem, Population, beta, rho, W, Z, i)
% Dimensionality reduction mechanism in DR-MCEA/D

%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Yuma Horaguchi

    PopObj     = Population.objs;
    PopDec     = Population.decs;

    %% Solution Sort & Selection
    PopTch     = max(abs(PopObj - repmat(Z, length(Population), 1)) .* W(i, :), [], 2);
    [~, Index] = sort(PopTch);
    Superior   = Index(1 : ceil(length(Population) * rho));
    Inferior   = Index(end - ceil(length(Population) * rho) + 1 : end);

    %% Statistics Mean Value
    Delta      = mean(PopDec(Superior, :), 1) - mean(PopDec(Inferior, :), 1); 
    DeltaSort  = sort(abs(Delta), 'descend');

    %% Selected beta * D Decision Variables
    IndexDif = find(abs(Delta) >= DeltaSort(ceil(beta * Problem.D)));
end