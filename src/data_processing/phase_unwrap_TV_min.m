function unwrapped_phase = phase_unwrap_TV_min(wrapped_phase,residue_add_check)

[m, n] = size(wrapped_phase); N_c = round(numel(wrapped_phase)*0.001);
unwrapped_phase_1st_mode = denoised_unwrap(wrapped_phase);
unwrapped_phase = unwrapped_phase_1st_mode;
number_of_iteration = 0;
while(1)
    phase_r = phase_wrap(wrapped_phase - unwrapped_phase); % Residual wrapped phase.
    % Counting numer of phase jump present in residual wrapped phase
    N = 0;
    for i = 1 : m
        for j = 1 : n
            if (abs(phase_r(i,j)) >= 2*pi)
                N = N + 1;
            end
        end
    end
    if (N < N_c)
        if (strcmp(residue_add_check,'yes') == 1)
            unwrapped_phase = unwrapped_phase + phase_r; % Eq.(11) of Ref.[1].
            break;
        else
            %Experience tells HN (High Noise) becomes blurry when final
            %residual wrapped phase is not added to output.
            break;
        end
    else
        unwrapped_phase = unwrapped_phase + denoised_unwrap(phase_r);
    end
    number_of_iteration = number_of_iteration + 1;
    if (number_of_iteration > 500)
        break;
    end
end
end

% H. Y. H. Huang, L. Tian, Z. Zhang, Y. Liu, Z. Chen, and G. Barbastathis, “Path-independent phase unwrapping using phase gradient and total-variation (TV) denoising,” Opt. Express, vol. 20, no. 13, p. 14075, Jun. 2012.
% ANTONIN CHAMBOLLE, “An Algorithm for Total Variation Minimization and Applications,” J. Math. Imaging Vis., vol. 20, no. 1/2, pp. 89?97, Jan. 2004.