%% PROPELLANT_SERVICE_LIFE_ML_LEBESGUE_CORRECTED.m
% =====================================================================
% ML PIPELINE WITH LEBESGUE-NORM LOADING -- DIMENSIONALLY CORRECT
% All stresses in ksc (kgf/cm^2).  1 MPa = 10.197162 ksc.
% =====================================================================
%
% KEY FIX from previous version:
%   The ML model is now trained on J_test -- the Lebesgue beta-norm of
%   each lab test's FULL stress-time history -- not on sigma_m (peak
%   stress).  This makes the comparison J_cycle / J_ref dimensionally
%   consistent: both sides are in ksc * s^(1/beta), so their ratio is
%   a pure number and the damage accumulation formula is physically
%   correct.
%
%   Previous version trained on sigma_m [ksc] and compared it to
%   J_cycle [ksc * s^(1/beta)], which is a units mismatch: the ratio
%   had hidden units of s^(1/beta), so (J_cycle/sigma_ref)^beta came
%   out in seconds rather than as a dimensionless damage fraction.
%   The ~10 year result was a numerical coincidence, not physics.
%
% eps_c (critical strain fraction) is NOT used in this model.
%   The damage integral gives time-to-failure directly.  eps_c was
%   relevant for the earlier master-curve / eps_dot route; it has no
%   role here, which is why changing it had no effect.
%
% =====================================================================
% INPUT FILES:
%   Lab UTM tests: T{T}_v{v}.xlsx  with columns [disp_mm, load_N, t_min]
%   Storage history: CSV with columns [time_s, sigma_ksc, T_K]
%     representing one full cycle of loading at the critical location.
% =====================================================================

clear; clc; close all;
rng(0);

%% =====================================================================
%%  PATHS -- EDIT THESE
%% =====================================================================
path_set1 = 'C:\Users\USER\Downloads\intern\set1';
path_set2 = 'C:\Users\USER\Downloads\intern\set2';
path_set3 = 'C:\Users\USER\Downloads\intern\set3';
stress_history_file = 'C:\Users\USER\Downloads\intern\storage_stress_history.csv';
output_file         = 'propellant_ml_lebesgue_corrected_results.csv';

%% 1. USER INPUTS ====================================================
L0   = 47.75;            % gauge length [mm]
A0   = 100;              % cross-sectional area [mm^2]

T_C  = [50, 60, 70];
v_mm = [5, 50, 500];

% Lebesgue exponent  (4-15 typical; higher -> peaks dominate)
beta = 8;

% Outlier exclusion
exclude_labels = {'T60_v500_set1'};

% Variance reduction by averaging replicates
average_replicates = true;

% Data folders
data_dirs = { path_set1, 'set1';
              path_set2, 'set2';
              path_set3, 'set3' };

% Unit conversion
MPa_to_ksc = 10.197162;      % 1 MPa = 10.197162 ksc

%% 2. DERIVED QUANTITIES =============================================
R       = 8.314;
T_K     = T_C + 273.15;
eps_dot = (v_mm/60) / L0;
nT = numel(T_C); nR = numel(v_mm);
nSets = size(data_dirs, 1);
nTests = nSets * nT * nR;

fprintf('Stress units: ksc throughout\n');
fprintf('Lebesgue exponent beta = %g\n\n', beta);

%% 3. LOAD DATA AND COMPUTE J_test FOR EACH LAB TEST =================
% KEY CHANGE: we keep the FULL stress-time history up to the peak
% and compute the Lebesgue beta-norm J_test from it.
% J_test = ( integral_0^{t_peak}  |sigma(t)|^beta  dt )^(1/beta)
% Units: ksc * s^(1/beta)
%
% This replaces sigma_m as the ML training target.
% sigma_m is still extracted for reference/display but NOT used in the
% damage model.

results = table('Size',[nTests 7], ...
    'VariableTypes',{'string','string','double','double','double','double','double'}, ...
    'VariableNames',{'label','set','T_K','eps_dot','sigma_m','eps_m','J_test'});

k = 0;
for s = 1:nSets
    dir_now = data_dirs{s,1};
    tag     = data_dirs{s,2};
    for i = 1:nT
        for j = 1:nR
            k = k + 1;
            label = sprintf('T%d_v%d_%s', T_C(i), v_mm(j), tag);
            fname = fullfile(dir_now, sprintf('T%d_v%d.xlsx', T_C(i), v_mm(j)));
            if ~isfile(fname)
                error('Missing file: %s', fname);
            end
            D = readmatrix(fname);
            ok      = ~any(isnan(D(:,1:3)), 2);
            D       = D(ok, :);
            disp_mm = D(:,1);
            load_N  = D(:,2);
            t_min   = D(:,3);

            sigma_ksc = (load_N / A0) * MPa_to_ksc;   % N/mm^2 = MPa * conv
            eps       = disp_mm / L0;
            t_sec     = t_min * 60;

            % Peak properties (for display only, not used in damage model)
            [sm, idx_m] = max(sigma_ksc);
            em = eps(idx_m);

            % ---- J_test: Lebesgue beta-norm up to peak ----
            % Only the loading portion (0 to t_peak) is used.
            % Post-peak softening is excluded -- it is not loading damage.
            sigma_to_peak = sigma_ksc(1:idx_m);
            t_to_peak     = t_sec(1:idx_m);

            integrand = abs(sigma_to_peak) .^ beta;
            I_test    = trapz(t_to_peak, integrand);   % ksc^beta * s
            J_test    = I_test ^ (1/beta);             % ksc * s^(1/beta)

            results(k,:) = {string(label), string(tag), T_K(i), eps_dot(j), sm, em, J_test};
        end
    end
end

fprintf('Loaded %d tests.\n', nTests);
fprintf('\n--- Raw J_test values (ksc * s^(1/beta)) ---\n');
disp(results(:, {'label','T_K','eps_dot','sigma_m','J_test'}));

%% 4. EXCLUDE OUTLIERS AND AVERAGE REPLICATES ========================
keep = ~ismember(results.label, exclude_labels);
data = results(keep,:);
if any(~keep)
    fprintf('Excluded: %s\n', strjoin(results.label(~keep), ', '));
end

if average_replicates
    [G, T_g, e_g] = findgroups(data.T_K, data.eps_dot);
    sm_avg    = splitapply(@mean, data.sigma_m, G);
    em_avg    = splitapply(@mean, data.eps_m,   G);
    J_avg     = splitapply(@mean, data.J_test,  G);   % <-- average J_test
    labels_g  = arrayfun(@(t,e) sprintf('T%d_v%d_avg', ...
        round(t-273.15), round(e*60*L0)), T_g, e_g, 'UniformOutput', false);
    data_used = table(string(labels_g), repmat("avg",numel(T_g),1), ...
        T_g, e_g, sm_avg, em_avg, J_avg, ...
        'VariableNames',{'label','set','T_K','eps_dot','sigma_m','eps_m','J_test'});
else
    data_used = data;
end
n = height(data_used);
fprintf('\nClean training set: %d points\n', n);
disp(data_used);

%% 5. FEATURE MATRIX =================================================
% Features: same physics-aware coordinates as before.
% TARGET (y): log( J_test )  -- NOT log(sigma_m) any more.
%
% Why log(J_test)?
%   J_test = (integral sigma^beta dt)^(1/beta)
%   In log space this is roughly linear in 1/T and log(eps_dot),
%   so the same feature engineering trick that worked for sigma_m
%   works for J_test too.

X = [1./data_used.T_K, log(data_used.eps_dot)];
y = log(data_used.J_test);           % <<< CORRECTED target

fprintf('\ny values being trained on = log(J_test).\n');
fprintf('J_test range: %.4f to %.4f ksc*s^(1/beta)\n', ...
    min(data_used.J_test), max(data_used.J_test));

%% 6. SIX-MODEL LOO COMPARISON ========================================
fprintf('\n----- Leave-One-Out CV (n=%d) -----\n', n);
fprintf('%-34s %-10s %-18s\n', 'Model', 'R^2(log)', 'RMSE(ksc*s^(1/b))');
fprintf('%s\n', repmat('-',1,64));

cv = struct();
cv.linear  = loocv(@(Xt,yt) fitlm(Xt,yt), X, y, 'Linear');
poly       = @(Z) [Z, Z.^2, Z(:,1).*Z(:,2)];
cv.poly    = loocvFeat(@(Xt,yt) fitlm(Xt,yt), poly, X, y, 'Polynomial deg-2');
cv.gpr_rbf = loocv(@(Xt,yt) fitrgp(Xt, yt, ...
    'KernelFunction','squaredexponential','BasisFunction','constant', ...
    'Standardize',true,'FitMethod','exact','PredictMethod','exact'), ...
    X, y, 'GPR (RBF)');
cv.gpr_mat = loocv(@(Xt,yt) fitrgp(Xt, yt, ...
    'KernelFunction','matern52','BasisFunction','constant', ...
    'Standardize',true,'FitMethod','exact','PredictMethod','exact'), ...
    X, y, 'GPR (Matern 5/2)');
cv.rf      = loocv(@(Xt,yt) fitrensemble(Xt, yt, 'Method','Bag', ...
    'NumLearningCycles',400,'Learners',templateTree('MaxNumSplits',3)), ...
    X, y, 'Random Forest');
cv.svm     = loocv(@(Xt,yt) fitrsvm(Xt, yt, 'KernelFunction','rbf', ...
    'Standardize',true,'KernelScale','auto'), X, y, 'SVM (RBF)');

fns   = fieldnames(cv);
rmses = cellfun(@(f) cv.(f).rmse_phys, fns);
[~, ib] = min(rmses);
best    = fns{ib};
fprintf('\nBest model by RMSE: %s\n', best);

%% 7. RETRAIN BEST + GPR FOR UNCERTAINTY =============================
switch best
    case 'linear';  mdl = fitlm(X, y);
                    predFn = @(Xq) predict(mdl, Xq);
    case 'poly';    mdl = fitlm(poly(X), y);
                    predFn = @(Xq) predict(mdl, poly(Xq));
    case 'gpr_rbf'; mdl = fitrgp(X, y, 'KernelFunction','squaredexponential', ...
                        'BasisFunction','constant','Standardize',true);
                    predFn = @(Xq) predict(mdl, Xq);
    case 'gpr_mat'; mdl = fitrgp(X, y, 'KernelFunction','matern52', ...
                        'BasisFunction','constant','Standardize',true);
                    predFn = @(Xq) predict(mdl, Xq);
    case 'rf';      mdl = fitrensemble(X, y, 'Method','Bag', ...
                        'NumLearningCycles',400, ...
                        'Learners',templateTree('MaxNumSplits',3));
                    predFn = @(Xq) predict(mdl, Xq);
    case 'svm';     mdl = fitrsvm(X, y, 'KernelFunction','rbf', ...
                        'Standardize',true,'KernelScale','auto');
                    predFn = @(Xq) predict(mdl, Xq);
end
gpr_uq = fitrgp(X, y, 'KernelFunction','matern52', ...
    'BasisFunction','constant', 'Standardize',true);

%% 8. READ STORAGE STRESS-TIME HISTORY ================================
if ~isfile(stress_history_file)
    error(['File not found: %s\n' ...
           'Expected CSV with columns: time_s, sigma_ksc, T_K'], ...
        stress_history_file);
end
H = readtable(stress_history_file);
if ~all(ismember({'time_s','sigma_ksc','T_K'}, H.Properties.VariableNames))
    H.Properties.VariableNames(1:3) = {'time_s','sigma_ksc','T_K'};
end
t_hist = H.time_s(:);
s_hist = H.sigma_ksc(:);           % ksc
T_hist = H.T_K(:);

T_cycle_mean    = trapz(t_hist, T_hist) / (t_hist(end) - t_hist(1));
cycle_dur_s     = t_hist(end) - t_hist(1);
cycle_dur_yr    = cycle_dur_s / (365.25*24*3600);

fprintf('\n--- Storage stress profile ---\n');
fprintf('  Cycle duration : %.4g s  (%.4f yr)\n', cycle_dur_s, cycle_dur_yr);
fprintf('  Stress range   : %.4f to %.4f ksc\n', min(s_hist), max(s_hist));
fprintf('  Mean stress    : %.4f ksc\n', mean(s_hist));
fprintf('  Mean T         : %.2f degC\n', T_cycle_mean - 273.15);

%% 9. J_cycle: LEBESGUE NORM OF SERVICE HISTORY =======================
% J_cycle = ( integral_0^{T_c}  |sigma_service(t)|^beta  dt )^(1/beta)
% Units: ksc * s^(1/beta)   -- SAME units as J_ref from ML.
I_cycle = trapz(t_hist, abs(s_hist).^beta);
J_cycle = I_cycle ^ (1/beta);                % ksc * s^(1/beta)

fprintf('\n--- Lebesgue norm of service history ---\n');
fprintf('  beta             = %g\n', beta);
fprintf('  J_cycle          = %.4e ksc*s^(1/beta)\n', J_cycle);

%% 10. J_ref FROM ML: REFERENCE NORM AT STORAGE CONDITIONS ===========
% Query the ML model at the SLOWEST TESTED strain rate and the
% cycle-averaged temperature.  The model now predicts log(J_test),
% so exp(prediction) is in ksc*s^(1/beta) -- same units as J_cycle.
edot_slow = min(eps_dot);
Xq_ref    = [1/T_cycle_mean, log(edot_slow)];
[logJ_ref, sd_ref] = predict(gpr_uq, Xq_ref);
J_ref     = exp(logJ_ref);                   % ksc * s^(1/beta) <<<
J_ref_lo  = exp(logJ_ref - sd_ref);
J_ref_hi  = exp(logJ_ref + sd_ref);

fprintf('\n--- J_ref from ML (T = %.2f degC, edot = %.3e 1/s) ---\n', ...
    T_cycle_mean-273.15, edot_slow);
fprintf('  J_ref          = %.4e ksc*s^(1/beta)\n', J_ref);
fprintf('  GPR -1sigma    = %.4e ksc*s^(1/beta)\n', J_ref_lo);
fprintf('  GPR +1sigma    = %.4e ksc*s^(1/beta)\n', J_ref_hi);
fprintf('  J_cycle/J_ref  = %.4f  (dimensionless; <1 means finite life)\n', ...
    J_cycle/J_ref);

%% 11. SERVICE LIFE -- CORRECTED FORMULA =============================
% D_per_cycle = (J_cycle / J_ref)^beta    [dimensionless]
% N_cycles    = 1 / D_per_cycle           [dimensionless]
% life        = N_cycles * cycle_duration [years]
%
% J_cycle and J_ref are both in ksc*s^(1/beta), so the ratio is
% dimensionless and the formula is units-consistent.

ratio       = J_cycle / J_ref;            % dimensionless
D_per_cycle = ratio ^ beta;              % dimensionless

fprintf('\n===== SERVICE LIFE (Lebesgue LCD, corrected) =====\n');
fprintf('  J_cycle / J_ref = %.4f\n', ratio);
fprintf('  (ratio)^beta    = %.4e  (damage per cycle)\n', D_per_cycle);

if ~isfinite(D_per_cycle) || D_per_cycle <= 0
    fprintf('  Damage per cycle is not finite -- check inputs.\n');
    life_yr = NaN;
elseif ratio >= 1
    % One cycle's load already exceeds the reference norm -- fails within
    % less than one cycle.
    fprintf(2,'  WARNING: J_cycle >= J_ref.  Propellant fails within 1 cycle.\n');
    life_yr = cycle_dur_yr * (J_ref/J_cycle)^beta;
    fprintf('  Predicted life : %.3e years\n', life_yr);
else
    N_cycles = 1 / D_per_cycle;
    life_yr  = N_cycles * cycle_dur_yr;
    fprintf('  Cycles to failure   : %.4e\n', N_cycles);
    fprintf('  >>> SERVICE LIFE    : %.2f years <<<\n', life_yr);
end

% Uncertainty from GPR band on J_ref
if isfinite(life_yr) && ~isnan(life_yr)
    D_lo   = (J_cycle / J_ref_hi)^beta;   % high J_ref -> less damage
    D_hi   = (J_cycle / J_ref_lo)^beta;   % low  J_ref -> more damage
    life_lo = cycle_dur_yr / D_hi;
    life_hi = cycle_dur_yr / D_lo;
    fprintf('  GPR band (1 sigma)  : %.2f  to  %.2f years\n', life_lo, life_hi);
end

%% 12. WHAT RATIO J_cycle/J_ref WOULD HIT 18-20 YEARS ================
% Useful for back-calculating whether your stress profile is in the
% right ballpark.
fprintf('\n--- Required J_cycle/J_ref for target lives ---\n');
fprintf('  %-12s  %-20s\n', 'Target (yr)', 'Required ratio J_c/J_ref');
for t_target = [5, 10, 15, 17, 18, 19, 20, 25]
    N_req   = t_target / cycle_dur_yr;
    D_req   = 1 / N_req;
    ratio_req = D_req ^ (1/beta);
    fprintf('  %-12g  %-20.4f\n', t_target, ratio_req);
end
fprintf('  Current J_cycle/J_ref = %.4f\n', ratio);

%% 13. CONTEXT: EQUIVALENT CONSTANT STRESS ===========================
sigma_equiv = J_cycle / cycle_dur_s^(1/beta);   % ksc
fprintf('\n--- Equivalent constant stress (context only, not used) ---\n');
fprintf('  %.5f ksc held for %.4f yr gives same Lebesgue norm\n', ...
    sigma_equiv, cycle_dur_yr);
fprintf('  = %.5f MPa = %.3f kPa\n', sigma_equiv/MPa_to_ksc, ...
    sigma_equiv/MPa_to_ksc*1e3);

%% 14. SENSITIVITY TO beta ===========================================
fprintf('\n--- Life vs beta (J_ref re-evaluated at each beta) ---\n');
fprintf('  %-8s %-16s %-16s %-16s %-14s\n', ...
    'beta','J_cycle','J_ref','ratio','life(yr)');

beta_grid  = [3 4 5 6 8 10 12 15];
life_grid  = nan(size(beta_grid));
for q = 1:numel(beta_grid)
    bq = beta_grid(q);

    % Recompute J_test for each test at this beta
    J_at_b = zeros(n, 1);
    for ki = 1:n
        % Re-derive J_test for averaged data at beta=bq
        % (Re-read files to get full history; average over sets)
        J_sum = 0; n_sum = 0;
        for s = 1:nSets
            T_now = data_used.T_K(ki);
            v_now = round(data_used.eps_dot(ki)*60*L0);
            lbl   = sprintf('T%d_v%d_%s', round(T_now-273.15), v_now, data_dirs{s,2});
            if ismember(string(lbl), exclude_labels), continue; end
            f = fullfile(data_dirs{s,1}, sprintf('T%d_v%d.xlsx', ...
                round(T_now-273.15), v_now));
            if ~isfile(f), continue; end
            Df = readmatrix(f);
            Df = Df(~any(isnan(Df(:,1:3)),2),:);
            sk = (Df(:,2)/A0)*MPa_to_ksc;
            tk = Df(:,3)*60;
            [~,im] = max(sk);
            Iq = trapz(tk(1:im), abs(sk(1:im)).^bq);
            J_sum = J_sum + Iq^(1/bq);
            n_sum = n_sum + 1;
        end
        if n_sum > 0, J_at_b(ki) = J_sum/n_sum; end
    end

    yq    = log(J_at_b);
    yq(~isfinite(yq)) = [];
    Xq_b  = X(isfinite(log(J_at_b)), :);
    if numel(yq) < 3, life_grid(q) = NaN; continue; end
    gq    = fitrgp(Xq_b, yq, 'KernelFunction','matern52', ...
                  'BasisFunction','constant','Standardize',true);
    [lJr, ~] = predict(gq, [1/T_cycle_mean, log(min(eps_dot))]);
    Jr_b  = exp(lJr);

    Ic_b  = trapz(t_hist, abs(s_hist).^bq);
    Jc_b  = Ic_b^(1/bq);

    r_b   = Jc_b / Jr_b;
    if r_b >= 1
        life_grid(q) = NaN;
    else
        life_grid(q) = cycle_dur_yr / r_b^bq;
    end
    fprintf('  %-8g %-16.4e %-16.4e %-16.4f %-14.3g\n', ...
        bq, Jc_b, Jr_b, r_b, life_grid(q));
end

%% 15. SAVE AND PLOT ==================================================
summary.best_model        = best;
summary.LOO_R2            = cv.(best).R2;
summary.LOO_RMSE_ksc_sb   = cv.(best).rmse_phys;
summary.n_training        = n;
summary.beta              = beta;
summary.J_cycle           = J_cycle;
summary.J_ref             = J_ref;
summary.ratio_Jc_Jr       = ratio;
summary.D_per_cycle       = D_per_cycle;
summary.service_life_yr   = life_yr;
if exist('life_lo','var')
    summary.life_band_lo  = life_lo;
    summary.life_band_hi  = life_hi;
end
disp(' '); disp('=== Summary ==='); disp(summary);

reduced = data_used;
reduced.J_test_ksc_sb = reduced.J_test;
writetable(reduced, output_file);
fprintf('\nOutput written to: %s\n', output_file);

% Plot 1: stress history
figure('Name','Storage stress history','Position',[80 80 1000 400]);
yyaxis left;  plot(t_hist/3600, s_hist, 'b-','LineWidth',1.2);
ylabel('Stress \sigma (ksc)');
yyaxis right; plot(t_hist/3600, T_hist-273.15, 'r--','LineWidth',1.2);
ylabel('Temperature (\circC)');
xlabel('Time (hours)'); grid on;
title(sprintf('Storage profile  (cycle = %.3f yr)', cycle_dur_yr));

% Plot 2: beta sensitivity on life
figure('Name','Beta sensitivity','Position',[120 120 900 500]);
valid = isfinite(life_grid);
semilogy(beta_grid(valid), life_grid(valid), 'bo-', ...
    'LineWidth',1.5,'MarkerSize',9,'MarkerFaceColor','b'); hold on;
yline(18,'k--','LineWidth',1.2,'Label','18 yr target');
yline(20,'k:','LineWidth',1.2,'Label','20 yr target');
xline(beta,'r--','LineWidth',1.5,'Label',sprintf('chosen \\beta=%g',beta));
xlabel('Lebesgue exponent \beta');
ylabel('Predicted service life (years)');
title('Life vs \beta  (corrected Lebesgue LCD)');
grid on;

% Plot 3: J_test values vs lab conditions
figure('Name','J_test from lab tests','Position',[160 160 800 500]);
colors = lines(nT); markers = {'o','s','^'};
eps_dot_unique = unique(data_used.eps_dot);
hold on;
for i = 1:nT
    mask = data_used.T_K == T_K(i);
    loglog(data_used.eps_dot(mask), data_used.J_test(mask), ...
        markers{i},'MarkerSize',11,'MarkerFaceColor',colors(i,:), ...
        'MarkerEdgeColor','k', ...
        'DisplayName',sprintf('%d \\circC',T_C(i)));
end
set(gca,'XScale','log','YScale','log');
xlabel('Strain rate \dot\epsilon (1/s)');
ylabel('J_{test}  (ksc \cdot s^{1/\beta})');
title(sprintf('Lebesgue \\beta-norm of lab tests (\\beta = %g)', beta));
legend('Location','best'); grid on;

%% ===================== LOCAL FUNCTIONS ============================
function out = loocv(trainFn, X, y, name)
    n = size(X,1); preds = zeros(n,1);
    for i = 1:n
        tr = setdiff(1:n,i);
        m  = trainFn(X(tr,:), y(tr));
        preds(i) = predict(m, X(i,:));
    end
    out.rmse_log  = sqrt(mean((y-preds).^2));
    out.rmse_phys = sqrt(mean((exp(y)-exp(preds)).^2));   % ksc*s^(1/b)
    out.R2        = 1 - sum((y-preds).^2)/sum((y-mean(y)).^2);
    out.preds     = preds;
    fprintf('%-34s %-10.4f %-18.5f\n', name, out.R2, out.rmse_phys);
end

function out = loocvFeat(trainFn, featFn, X, y, name)
    n = size(X,1); preds = zeros(n,1);
    for i = 1:n
        tr = setdiff(1:n,i);
        m  = trainFn(featFn(X(tr,:)), y(tr));
        preds(i) = predict(m, featFn(X(i,:)));
    end
    out.rmse_log  = sqrt(mean((y-preds).^2));
    out.rmse_phys = sqrt(mean((exp(y)-exp(preds)).^2));
    out.R2        = 1 - sum((y-preds).^2)/sum((y-mean(y)).^2);
    out.preds     = preds;
    fprintf('%-34s %-10.4f %-18.5f\n', name, out.R2, out.rmse_phys);
end
