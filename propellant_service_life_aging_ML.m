%% PROPELLANT_SERVICE_LIFE_AGING_ML.m
% =====================================================================
% SERVICE LIFE PREDICTION FROM ACCELERATED AGING DATA (~500 FILES)
% =====================================================================
%
% FILE NAMING CONVENTION:
%   T{T}_d{d}_v{v}_s{s}.xlsx
%   T = aging temperature     [degC]  e.g. 50, 60, 70
%   d = days aged at T        [days]  e.g. 0, 7, 14, 30, 60, 90 ...
%   v = test crosshead speed  [mm/min] 5, 50, or 500
%   s = sample/replicate no.
%   Columns in each file: disp_mm | load_N | t_min
%
% WHAT THIS SCRIPT DOES
%   1.  Reads every .xlsx file in data_dir, parses T/d/v/s from name.
%   2.  Extracts peak stress sigma_m and strain eps_m from each test.
%   3.  Averages replicates at the same (T, d, v) condition.
%   4.  Fits an Arrhenius activation energy Ea for CHEMICAL DEGRADATION
%       by finding the Ea that best collapses the sigma_m vs aging-time
%       curves from all three temperatures onto one master curve.
%   5.  Computes equivalent aging time at service temperature:
%         d_equiv = d × exp( Ea/R × (1/T_service - 1/T_aging) )
%   6.  Trains 6 ML models on [d_equiv, log(eps_dot)] → sigma_m.
%       Picks the best by 5-fold cross-validation.
%   7.  Predicts service life = equivalent time for sigma_m to drop
%       to (threshold_frac × initial strength) at T_service.
%
% IMPORTANT NOTES
%   - Ea here is the CHEMICAL AGING activation energy, NOT the
%     mechanical (TTRS) Ea used in earlier scripts. They are different.
%     Chemical aging Ea for HTPB-AP is typically 80-130 kJ/mol.
%   - The threshold (default 0.8 = 20% strength loss) is the criterion
%     for "end of service life". Change threshold_frac to suit your
%     requirement.
%   - sigma_m0 (initial strength) is the ML prediction at d_equiv = 0.
%     If d=0 files exist in your dataset, the model is anchored on them.
%     If not, the model extrapolates from the earliest aged data.
%
% STRESS UNITS: ksc (kgf/cm^2).  1 MPa = 10.197162 ksc.
% =====================================================================

clear; clc; close all;
rng(0);

%% =====================================================================
%% 1. USER INPUTS — EDIT THESE
%% =====================================================================
data_dir      = 'C:\Users\USER\Downloads\intern\DATA';  % folder with all .xlsx files
output_csv    = 'aging_service_life_results.csv';

L0 = 47.75;             % gauge length [mm]
A0 = 25;                % cross-sectional area [mm^2]  <-- confirmed by user

T_service_C   = 27;     % service/storage temperature [degC]
threshold_frac = 0.8;   % failure = strength drops to this fraction of initial
                        % 0.8 = 20% strength loss (typical propellant criterion)
                        % change to 0.5 for 50% loss if that suits your spec

v_ref_mmmin   = 5;      % reference test speed for service life curve [mm/min]
                        % use slowest (5) for conservative prediction

MPa_to_ksc    = 10.197162;

%% =====================================================================
%% 2. SCAN DIRECTORY AND PARSE FILENAMES
%% =====================================================================
files = dir(fullfile(data_dir, 'T*_d*_v*_s*.xlsx'));
if isempty(files)
    error(['No files matching T*_d*_v*_s*.xlsx found in:\n  %s\n' ...
           'Check data_dir and that filenames match the pattern.'], data_dir);
end
fprintf('Found %d files.\n', numel(files));

nFiles = numel(files);
T_age_all  = nan(nFiles,1);
d_age_all  = nan(nFiles,1);
v_all      = nan(nFiles,1);
s_all      = nan(nFiles,1);
fnames     = cell(nFiles,1);

for k = 1:nFiles
    fnames{k} = files(k).name;
    tok = regexp(files(k).name, ...
                 'T(\d+)_d(\d+)_v(\d+)_s(\d+)', 'tokens');
    if isempty(tok)
        warning('Cannot parse filename: %s -- skipping.', files(k).name);
        continue;
    end
    T_age_all(k) = str2double(tok{1}{1});
    d_age_all(k) = str2double(tok{1}{2});
    v_all(k)     = str2double(tok{1}{3});
    s_all(k)     = str2double(tok{1}{4});
end

% Drop any rows that failed to parse
valid = ~isnan(T_age_all);
T_age_all = T_age_all(valid);
d_age_all = d_age_all(valid);
v_all     = v_all(valid);
s_all     = s_all(valid);
fnames    = fnames(valid);
nFiles    = sum(valid);
fprintf('Parsed %d valid filenames.\n', nFiles);

unique_T_age = unique(T_age_all);
unique_d     = unique(d_age_all);
unique_v     = unique(v_all);
fprintf('Aging temperatures : %s degC\n', mat2str(unique_T_age'));
fprintf('Aging durations    : %s days\n', mat2str(unique_d'));
fprintf('Test speeds        : %s mm/min\n\n', mat2str(unique_v'));

%% =====================================================================
%% 3. READ EVERY FILE AND EXTRACT sigma_m, eps_m
%% =====================================================================
R_gas     = 8.314;
T_K_all   = T_age_all + 273.15;
T_service_K = T_service_C + 273.15;
eps_dot_all = (v_all / 60) / L0;   % strain rates [1/s]

sigma_m_all = nan(nFiles,1);
eps_m_all   = nan(nFiles,1);

fprintf('Reading files ');
for k = 1:nFiles
    if mod(k, 50) == 0, fprintf('%d...', k); end
    fpath = fullfile(data_dir, fnames{k});
    try
        D = readmatrix(fpath);
        % Drop any NaN rows
        D = D(~any(isnan(D(:,1:3)), 2), :);
        if size(D,1) < 3, continue; end

        disp_mm = D(:,1);
        load_N  = D(:,2);
        sigma_ksc = (load_N / A0) * MPa_to_ksc;  % ksc
        eps       = disp_mm / L0;

        [sm, im]       = max(sigma_ksc);
        sigma_m_all(k) = sm;
        eps_m_all(k)   = eps(im);
    catch
        warning('Could not read: %s', fnames{k});
    end
end
fprintf(' done.\n\n');

% Remove files that could not be read
ok = ~isnan(sigma_m_all);
T_K_all     = T_K_all(ok);
T_age_all   = T_age_all(ok);
d_age_all   = d_age_all(ok);
v_all       = v_all(ok);
s_all       = s_all(ok);
eps_dot_all = eps_dot_all(ok);
sigma_m_all = sigma_m_all(ok);
eps_m_all   = eps_m_all(ok);
fnames      = fnames(ok);
nValid      = sum(ok);
fprintf('Successfully read %d / %d files.\n', nValid, nFiles);

%% =====================================================================
%% 4. PRINT RAW DATA SUMMARY
%% =====================================================================
fprintf('\n--- Raw sigma_m range (ksc) by aging temperature ---\n');
for Ti = unique_T_age'
    mask = T_age_all == Ti;
    fprintf('  T = %3d degC : min=%.4f  max=%.4f  mean=%.4f  n=%d\n', ...
        Ti, min(sigma_m_all(mask)), max(sigma_m_all(mask)), ...
        mean(sigma_m_all(mask)), sum(mask));
end

%% =====================================================================
%% 5. AVERAGE REPLICATES
%% Keep one value per unique (T_aging, d_days, v_speed) combination.
%% =====================================================================
T_K_vec    = [];
d_vec      = [];
eps_dot_vec= [];
sigma_m_vec= [];
eps_m_vec  = [];
T_age_vec  = [];

conds = unique([T_age_all, d_age_all, v_all], 'rows');
for c = 1:size(conds,1)
    mask = T_age_all == conds(c,1) & ...
           d_age_all == conds(c,2) & ...
           v_all     == conds(c,3);
    if sum(mask) == 0, continue; end
    T_K_vec     = [T_K_vec;     conds(c,1)+273.15];     %#ok<AGROW>
    T_age_vec   = [T_age_vec;   conds(c,1)];             %#ok<AGROW>
    d_vec       = [d_vec;       conds(c,2)];             %#ok<AGROW>
    eps_dot_vec = [eps_dot_vec; mean(eps_dot_all(mask))];%#ok<AGROW>
    sigma_m_vec = [sigma_m_vec; mean(sigma_m_all(mask))];%#ok<AGROW>
    eps_m_vec   = [eps_m_vec;   mean(eps_m_all(mask))];  %#ok<AGROW>
end
n = numel(sigma_m_vec);
fprintf('Averaged to %d unique (T, d, v) conditions.\n\n', n);

%% =====================================================================
%% 6. FIT ARRHENIUS Ea FOR CHEMICAL DEGRADATION
%% =====================================================================
% Find Ea that best collapses sigma_m vs aging-time curves at different
% temperatures onto one master degradation curve.
% 
% Objective: fit a degree-2 polynomial in log(d_equiv+1) to sigma_m,
%            minimise the sum of squared residuals.
% Ea bounded to [40, 200] kJ/mol -- physically reasonable for HTPB-AP.
% Negative Ea is not permitted (degradation MUST be faster at higher T).

fprintf('Fitting Arrhenius Ea for chemical degradation...\n');

obj_Ea = @(Ea_kJ) aging_scatter(Ea_kJ, T_K_vec, d_vec, ...
                                 sigma_m_vec, T_service_K, R_gas);

Ea_opt = fminbnd(obj_Ea, 40, 200, optimset('TolX', 1e-4));
scatter_val = obj_Ea(Ea_opt);

fprintf('  Ea (chemical aging) = %.2f kJ/mol\n', Ea_opt);
fprintf('  Residual scatter     = %.5f\n\n', scatter_val);
if Ea_opt < 45
    fprintf(2,'  Note: low Ea (<45 kJ/mol). Check that aging temperatures\n');
    fprintf(2,'  span a wide enough range and that d=0 (fresh) data exists.\n\n');
end

%% =====================================================================
%% 7. COMPUTE EQUIVALENT AGING TIME AT T_SERVICE
%% =====================================================================
% d_equiv = d × exp(Ea/R × (1/T_service - 1/T_aging))
% Acceleration factors for each aging temperature:
Ea = Ea_opt * 1e3;
accel_factors = exp((Ea/R_gas) .* (1/T_service_K - 1./T_K_vec));
d_equiv_vec   = d_vec .* accel_factors;   % equivalent days at T_service

fprintf('--- Arrhenius acceleration factors ---\n');
for Ti = unique_T_age'
    mask = T_age_vec == Ti;
    if any(mask)
        af = mean(accel_factors(mask));
        fprintf('  %3d degC -> 1 day = %.1f equiv. days at %.0f degC\n', ...
            Ti, af, T_service_C);
    end
end
fprintf('  --> %.0f days at T_aging maps to ~%.1f years at %.0f degC\n\n', ...
    max(d_vec), max(d_equiv_vec)/365.25, T_service_C);

%% =====================================================================
%% 8. FEATURE MATRIX AND ML TRAINING
%% =====================================================================
% Features: [log(d_equiv+1),  log(eps_dot)]
%   log(d_equiv+1) handles d=0 gracefully (log(1)=0)
%   log(eps_dot)   captures rate-dependence in the mechanical test
%
% Target: sigma_m [ksc]
% Using log(sigma_m) as ML target makes residuals more symmetric.

X = [log(d_equiv_vec + 1),  log(eps_dot_vec)];
y = log(sigma_m_vec);

fprintf('----- 5-Fold Cross-Validation (n=%d) -----\n', n);
fprintf('%-34s %-10s %-12s\n', 'Model', 'R^2', 'RMSE(ksc)');
fprintf('%s\n', repmat('-', 1, 58));

cv_folds = 5;
poly2 = @(Z) [Z, Z.^2, Z(:,1).*Z(:,2)];

cv.linear  = kfold_cv(@(Xt,yt) fitlm(Xt, yt), X, y, cv_folds, ...
                       'Linear');
cv.poly2   = kfold_cv_feat(@(Xt,yt) fitlm(Xt, yt), poly2, X, y, ...
                       cv_folds, 'Polynomial (deg 2)');
cv.gpr_mat = kfold_cv(@(Xt,yt) fitrgp(Xt, yt, ...
                       'KernelFunction','matern52', ...
                       'BasisFunction','linear', ...
                       'Standardize', true), X, y, cv_folds, 'GPR Matern 5/2');
cv.gpr_rbf = kfold_cv(@(Xt,yt) fitrgp(Xt, yt, ...
                       'KernelFunction','squaredexponential', ...
                       'BasisFunction','linear', ...
                       'Standardize', true), X, y, cv_folds, 'GPR RBF');
cv.rf      = kfold_cv(@(Xt,yt) fitrensemble(Xt, yt, 'Method','Bag', ...
                       'NumLearningCycles', 300, ...
                       'Learners', templateTree('MaxNumSplits', 5)), ...
                       X, y, cv_folds, 'Random Forest');
cv.svm     = kfold_cv(@(Xt,yt) fitrsvm(Xt, yt, ...
                       'KernelFunction','rbf', 'Standardize', true, ...
                       'KernelScale','auto'), X, y, cv_folds, 'SVM RBF');

fns   = fieldnames(cv);
rmses = cellfun(@(f) cv.(f).rmse_ksc, fns);
[~, ib] = min(rmses);
best    = fns{ib};
fprintf('\nBest model: %s  (RMSE = %.5f ksc)\n\n', best, rmses(ib));

%% =====================================================================
%% 9. RETRAIN BEST MODEL ON FULL DATA + GPR FOR UNCERTAINTY
%% =====================================================================
switch best
    case 'linear'
        mdl    = fitlm(X, y);
        predFn = @(Xq) predict(mdl, Xq);
    case 'poly2'
        mdl    = fitlm(poly2(X), y);
        predFn = @(Xq) predict(mdl, poly2(Xq));
    case 'gpr_mat'
        mdl    = fitrgp(X, y, 'KernelFunction','matern52', ...
                        'BasisFunction','linear', 'Standardize',true);
        predFn = @(Xq) predict(mdl, Xq);
    case 'gpr_rbf'
        mdl    = fitrgp(X, y, 'KernelFunction','squaredexponential', ...
                        'BasisFunction','linear', 'Standardize',true);
        predFn = @(Xq) predict(mdl, Xq);
    case 'rf'
        mdl    = fitrensemble(X, y, 'Method','Bag', ...
                              'NumLearningCycles',300, ...
                              'Learners', templateTree('MaxNumSplits',5));
        predFn = @(Xq) predict(mdl, Xq);
    case 'svm'
        mdl    = fitrsvm(X, y, 'KernelFunction','rbf', ...
                         'Standardize',true, 'KernelScale','auto');
        predFn = @(Xq) predict(mdl, Xq);
end

% GPR always co-trained for uncertainty bounds
gpr_uq = fitrgp(X, y, 'KernelFunction','matern52', ...
                'BasisFunction','linear', 'Standardize',true);

%% =====================================================================
%% 10. SERVICE LIFE PREDICTION
%% =====================================================================
% Sweep equivalent aging time from 0 to 100 years at T_service.
% Use reference strain rate v_ref for the mechanical test.

eps_dot_ref   = (v_ref_mmmin / 60) / L0;
d_equiv_sweep = linspace(0, 100*365.25, 5000);    % equiv. days at T_service

Xq_sweep  = [log(d_equiv_sweep(:) + 1), ...
              repmat(log(eps_dot_ref), numel(d_equiv_sweep), 1)];

% Best model prediction
log_sig_sweep         = predFn(Xq_sweep);
sigma_m_sweep         = exp(log_sig_sweep);

% GPR prediction + uncertainty
[log_sig_gpr, sd_gpr] = predict(gpr_uq, Xq_sweep);
sigma_gpr             = exp(log_sig_gpr);
sigma_gpr_lo          = exp(log_sig_gpr - sd_gpr);
sigma_gpr_hi          = exp(log_sig_gpr + sd_gpr);

% Initial strength (d_equiv = 0, fresh material)
sigma_m0      = exp(predFn([0, log(eps_dot_ref)]));
sigma_thresh  = threshold_frac * sigma_m0;

fprintf('=== Service life prediction ===\n');
fprintf('  v_ref          : %.0f mm/min\n', v_ref_mmmin);
fprintf('  sigma_m0       : %.4f ksc  (= %.4f MPa)\n', ...
        sigma_m0, sigma_m0/MPa_to_ksc);
fprintf('  threshold (%.0f%%): %.4f ksc\n', threshold_frac*100, sigma_thresh);

% Find crossing point
cross_idx = find(sigma_m_sweep <= sigma_thresh, 1, 'first');

if isempty(cross_idx)
    life_yr  = Inf;
    life_txt = sprintf('> %.0f years (threshold not reached)', ...
                       d_equiv_sweep(end)/365.25);
    fprintf('  >>> No threshold crossing in %.0f years. Life > %.0f yr\n', ...
            d_equiv_sweep(end)/365.25, d_equiv_sweep(end)/365.25);
else
    % Interpolate for precision
    if cross_idx > 1
        x1 = d_equiv_sweep(cross_idx-1); y1 = sigma_m_sweep(cross_idx-1);
        x2 = d_equiv_sweep(cross_idx);   y2 = sigma_m_sweep(cross_idx);
        d_life = x1 + (sigma_thresh - y1) * (x2 - x1) / (y2 - y1);
    else
        d_life = d_equiv_sweep(cross_idx);
    end
    life_yr  = d_life / 365.25;
    life_txt = sprintf('%.2f years', life_yr);

    % GPR uncertainty on life
    cross_lo = find(sigma_gpr_hi <= sigma_thresh, 1, 'first');
    cross_hi = find(sigma_gpr_lo <= sigma_thresh, 1, 'first');
    life_lo  = [];  life_hi = [];
    if ~isempty(cross_lo), life_lo = d_equiv_sweep(cross_lo)/365.25; end
    if ~isempty(cross_hi), life_hi = d_equiv_sweep(cross_hi)/365.25; end

    fprintf('\n  ============================================================\n');
    fprintf('  PREDICTED SERVICE LIFE: %.2f years at %.0f degC\n', ...
            life_yr, T_service_C);
    fprintf('  (equivalent aging time to %.0f%% strength retention)\n', ...
            threshold_frac*100);
    if ~isempty(life_lo) && ~isempty(life_hi)
        fprintf('  GPR uncertainty band  : %.2f -- %.2f years (±1 sigma)\n', ...
                life_lo, life_hi);
    end
    fprintf('  ============================================================\n\n');
end

%% =====================================================================
%% 11. SENSITIVITY TABLES
%% =====================================================================

% A: sensitivity to threshold
fprintf('--- Life vs strength-retention threshold ---\n');
fprintf('  %-20s %-14s %-14s\n', 'Threshold (%)', 'sigma (ksc)', 'life (yr)');
for tf = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    st = tf * sigma_m0;
    ci = find(sigma_m_sweep <= st, 1, 'first');
    if isempty(ci)
        lv = Inf;
    else
        if ci > 1
            x1 = d_equiv_sweep(ci-1); y1 = sigma_m_sweep(ci-1);
            x2 = d_equiv_sweep(ci);   y2 = sigma_m_sweep(ci);
            lv = (x1 + (st-y1)*(x2-x1)/(y2-y1)) / 365.25;
        else
            lv = d_equiv_sweep(ci)/365.25;
        end
    end
    marker = '';
    if abs(tf - threshold_frac) < 1e-9, marker = '  <-- chosen'; end
    fprintf('  %-20.0f %-14.4f %-14.4g%s\n', tf*100, st, lv, marker);
end

% B: sensitivity to test strain rate (v)
fprintf('\n--- Life vs test strain rate (v_ref) ---\n');
fprintf('  %-14s %-14s\n', 'v_ref (mm/min)', 'life (yr)');
for vq = unique_v'
    edq    = (vq/60)/L0;
    Xqv    = [log(d_equiv_sweep(:)+1), repmat(log(edq), numel(d_equiv_sweep),1)];
    sm_v   = exp(predFn(Xqv));
    sm0_v  = sm_v(1);
    thr_v  = threshold_frac * sm0_v;
    ci     = find(sm_v <= thr_v, 1, 'first');
    lv_v   = Inf;
    if ~isempty(ci) && ci > 1
        x1 = d_equiv_sweep(ci-1); y1 = sm_v(ci-1);
        x2 = d_equiv_sweep(ci);   y2 = sm_v(ci);
        lv_v = (x1 + (thr_v-y1)*(x2-x1)/(y2-y1))/365.25;
    end
    marker = '';
    if vq == v_ref_mmmin, marker = '  <-- chosen'; end
    fprintf('  %-14.0f %-14.4g%s\n', vq, lv_v, marker);
end

% C: sensitivity to assumed Ea
fprintf('\n--- Life vs Ea assumption (chemical aging Ea) ---\n');
fprintf('  %-14s %-14s %-14s\n', 'Ea (kJ/mol)', 'd_equiv_max (yr)', 'life (yr)');
for Eq = [60 70 80 90 100 110 120 130 140]
    af_q    = exp((Eq*1e3/R_gas).*(1/T_service_K - 1./T_K_vec));
    deq_q   = d_vec .* af_q;
    Xq_tr   = [log(deq_q+1), log(eps_dot_vec)];
    yq      = log(sigma_m_vec);
    gq      = fitrgp(Xq_tr, yq, 'KernelFunction','matern52', ...
                    'BasisFunction','linear','Standardize',true);
    sm_q    = exp(predict(gq, [log(d_equiv_sweep(:)+1), ...
                  repmat(log(eps_dot_ref), numel(d_equiv_sweep),1)]));
    sm0_q   = sm_q(1);
    thr_q   = threshold_frac * sm0_q;
    ci      = find(sm_q <= thr_q, 1, 'first');
    lv_q    = Inf;
    if ~isempty(ci) && ci > 1
        x1 = d_equiv_sweep(ci-1); y1 = sm_q(ci-1);
        x2 = d_equiv_sweep(ci);   y2 = sm_q(ci);
        lv_q = (x1+(thr_q-y1)*(x2-x1)/(y2-y1))/365.25;
    end
    marker = '';
    if abs(Eq - Ea_opt) < 5, marker = '  <-- fitted'; end
    fprintf('  %-14.0f %-14.2f %-14.4g%s\n', Eq, ...
            max(deq_q)/365.25, lv_q, marker);
end

%% =====================================================================
%% 12. PLOTS
%% =====================================================================

% ---- Plot 1: Raw degradation curves (sigma_m vs d_days by temperature)
figure('Name','Raw degradation', 'Position', [60 60 1000 550]);
colors = lines(numel(unique_T_age));
markers_list = {'o','s','^','d','v'};
for ti = 1:numel(unique_T_age)
    T_now = unique_T_age(ti);
    for vi = 1:numel(unique_v)
        v_now = unique_v(vi);
        mask  = T_age_vec == T_now & eps_dot_vec*60*L0 == v_now;
        if sum(mask) < 2, continue; end
        [ds, srt] = sort(d_vec(mask));
        sm_sorted = sigma_m_vec(mask); sm_sorted = sm_sorted(srt);
        plot(ds, sm_sorted, ['-' markers_list{vi}], ...
            'Color', colors(ti,:), 'MarkerFaceColor', colors(ti,:), ...
            'LineWidth', 1.2, 'MarkerSize', 7, ...
            'DisplayName', sprintf('%d\\circC, v=%d mm/min', T_now, v_now));
        hold on;
    end
end
xlabel('Aging duration (days)'); ylabel('\sigma_m (ksc)');
title('Degradation of peak strength vs aging time');
legend('Location', 'best', 'NumColumns', 2);
grid on;

% ---- Plot 2: Collapsed master curve (sigma_m vs d_equiv)
figure('Name','Master degradation curve', 'Position', [100 100 900 550]);
for ti = 1:numel(unique_T_age)
    T_now = unique_T_age(ti);
    mask  = T_age_vec == T_now;
    [de_s, srt] = sort(d_equiv_vec(mask));
    sm_s = sigma_m_vec(mask); sm_s = sm_s(srt);
    semilogx(de_s + 1, sm_s, ['-' markers_list{ti}], ...
        'Color', colors(ti,:), 'MarkerFaceColor', colors(ti,:), ...
        'LineWidth', 1.2, 'MarkerSize', 8, ...
        'DisplayName', sprintf('%d\\circC (Arrhenius-shifted)', T_now));
    hold on;
end
% overlay ML fit
d_fit_sweep = logspace(0, log10(100*365.25+1), 200) - 1;
for vi = 1:numel(unique_v)
    vq = unique_v(vi); edq = (vq/60)/L0;
    Xqf = [log(d_fit_sweep(:)+1), repmat(log(edq), numel(d_fit_sweep),1)];
    sm_f = exp(predFn(Xqf));
    semilogx(d_fit_sweep+1, sm_f, '--', 'LineWidth', 1.6, ...
        'Color', [0 0 0]+0.3*vi/numel(unique_v), ...
        'DisplayName', sprintf('ML fit, v=%d mm/min', vq));
end
xlabel('Equivalent aging time at 27\circC + 1 (days)');
ylabel('\sigma_m (ksc)');
title(sprintf('Arrhenius-collapsed master degradation curve (E_a = %.1f kJ/mol)', Ea_opt));
legend('Location', 'best'); grid on;

% ---- Plot 3: Predicted vs Actual
figure('Name','Predicted vs Actual', 'Position', [140 140 700 600]);
y_pred_cv = cv.(best).y_pred_cv;
scatter(exp(y), exp(y_pred_cv), 60, d_vec, 'filled');
hold on; ref = [min(sigma_m_vec)*0.9, max(sigma_m_vec)*1.1];
plot(ref, ref, 'k--', 'LineWidth', 1.2);
xlabel('Measured \sigma_m (ksc)'); ylabel('Predicted \sigma_m (ksc)');
title(sprintf('Predicted vs Actual  (%s, R^2 = %.3f)', best, cv.(best).R2));
c = colorbar; c.Label.String = 'Aging days';
grid on;

% ---- Plot 4: Service life curve with uncertainty
figure('Name','Service life', 'Position', [180 180 1000 580]);
d_yr = d_equiv_sweep / 365.25;
fill([d_yr, fliplr(d_yr)], ...
     [sigma_gpr_lo', fliplr(sigma_gpr_hi')], ...
     [0.6 0.8 1.0], 'FaceAlpha', 0.3, 'EdgeColor', 'none', ...
     'DisplayName', '\pm1\sigma (GPR)');
hold on;
plot(d_yr, sigma_gpr,      'b-',  'LineWidth', 2, 'DisplayName', 'GPR mean');
plot(d_yr, sigma_m_sweep,  'r--', 'LineWidth', 1.5, ...
    'DisplayName', sprintf('%s (best model)', best));
yline(sigma_thresh, 'k-', 'LineWidth', 1.5, ...
    'Label', sprintf('%.0f%% threshold (%.4f ksc)', threshold_frac*100, sigma_thresh));
yline(sigma_m0, 'k:', 'LineWidth', 1, ...
    'Label', sprintf('\\sigma_{m0} = %.4f ksc', sigma_m0));
if isfinite(life_yr)
    xline(life_yr, 'm--', 'LineWidth', 1.5, ...
        'Label', sprintf('Life = %.1f yr', life_yr));
end
xlabel('Equivalent aging time at T_{service} = 27\circC (years)');
ylabel('\sigma_m (ksc)');
title(sprintf('Predicted degradation at v = %.0f mm/min', v_ref_mmmin));
legend('Location', 'northeast'); grid on; xlim([0 min(100, 2*life_yr+10)]);

%% =====================================================================
%% 13. SAVE RESULTS
%% =====================================================================
summary.Ea_kJ             = Ea_opt;
summary.n_conditions      = n;
summary.best_model        = best;
summary.CV_R2             = cv.(best).R2;
summary.CV_RMSE_ksc       = cv.(best).rmse_ksc;
summary.sigma_m0_ksc      = sigma_m0;
summary.threshold_frac    = threshold_frac;
summary.sigma_thresh_ksc  = sigma_thresh;
summary.service_life_yr   = life_yr;
if exist('life_lo','var') && ~isempty(life_lo)
    summary.life_lo_yr    = life_lo;
    summary.life_hi_yr    = life_hi;
end
disp(' '); disp('=== Final summary ==='); disp(summary);

% Save averaged dataset with d_equiv
out_tbl = table(T_age_vec, d_vec, round(eps_dot_vec*60*L0), ...
    d_equiv_vec, sigma_m_vec, eps_m_vec, ...
    'VariableNames', {'T_aging_degC','d_days','v_mmmin', ...
                      'd_equiv_days','sigma_m_ksc','eps_m'});
writetable(out_tbl, output_csv);
fprintf('\nData table saved to: %s\n', output_csv);

%% =====================================================================
%% LOCAL FUNCTIONS
%% =====================================================================
function s = aging_scatter(Ea_kJ, T_K_vec, d_vec, sigma_m_vec, T_service_K, R)
% Scatter of sigma_m about a polynomial fit in d_equiv space.
% This is the objective minimised to find Ea.
    Ea     = Ea_kJ * 1e3;
    d_eq   = d_vec .* exp((Ea/R) .* (1/T_service_K - 1./T_K_vec));
    x      = log(d_eq + 1);
    p      = polyfit(x, sigma_m_vec, 2);
    s      = sum((sigma_m_vec - polyval(p, x)).^2);
end

function out = kfold_cv(trainFn, X, y, k, name)
    n    = size(X,1);
    idx  = crossvalind('Kfold', n, k);
    pred = zeros(n,1);
    for f = 1:k
        te = idx == f; tr = ~te;
        m  = trainFn(X(tr,:), y(tr));
        pred(te) = predict(m, X(te,:));
    end
    out.y_pred_cv = pred;
    out.rmse_ksc  = sqrt(mean((exp(y) - exp(pred)).^2));
    out.R2        = 1 - sum((y-pred).^2) / sum((y-mean(y)).^2);
    fprintf('%-34s %-10.4f %-12.5f\n', name, out.R2, out.rmse_ksc);
end

function out = kfold_cv_feat(trainFn, featFn, X, y, k, name)
    n    = size(X,1);
    idx  = crossvalind('Kfold', n, k);
    pred = zeros(n,1);
    for f = 1:k
        te = idx == f; tr = ~te;
        m  = trainFn(featFn(X(tr,:)), y(tr));
        pred(te) = predict(m, featFn(X(te,:)));
    end
    out.y_pred_cv = pred;
    out.rmse_ksc  = sqrt(mean((exp(y) - exp(pred)).^2));
    out.R2        = 1 - sum((y-pred).^2) / sum((y-mean(y)).^2);
    fprintf('%-34s %-10.4f %-12.5f\n', name, out.R2, out.rmse_ksc);
end
