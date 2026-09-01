%% plot_robustness_paper.m
% Publication-oriented figures for PPO residual landing robustness evaluation.
%
% Required files:
%   runs/paper_eval/robustness_summary.csv
%   runs/paper_eval/robustness_episodes.csv
%   runs/paper_eval/robustness_paired_pid_vs_ppo.csv
%
% Output:
%   runs/paper_eval/fig_robustness_success.pdf
%   runs/paper_eval/fig_robustness_success.png
%   runs/paper_eval/fig_mixed_detailed.pdf
%   runs/paper_eval/fig_mixed_detailed.png
%   runs/paper_eval/fig_paired_rescue.pdf
%   runs/paper_eval/fig_paired_rescue.png
%
% Tested conceptually for MATLAB R2020b+.
% If exportgraphics is unavailable, replace exportgraphics with print/saveas.

clear; clc; close all;

%% ------------------------------------------------------------------------
%  User settings
% -------------------------------------------------------------------------
dataDir = fullfile("runs", "paper_eval");

summaryFile  = fullfile(dataDir, "robustness_summary.csv");
episodeFile  = fullfile(dataDir, "robustness_episodes.csv");
pairedFile   = fullfile(dataDir, "robustness_paired_pid_vs_ppo.csv");

fontName = "Times New Roman";
fontSize = 9;
lineWidth = 1.25;
markerSize = 6;

% Keep this TRUE for the paper-ready robustness figure.
showDeltaLabels = true;

% Figure size is intentionally compact for a two-column conference paper.
successFigSizeCm = [17.0, 7.2];   % width, height
mixedFigSizeCm   = [17.0, 6.5];
pairedFigSizeCm  = [17.0, 6.5];

%% ------------------------------------------------------------------------
%  Load data
% -------------------------------------------------------------------------
assert(isfile(summaryFile), "Missing file: %s", summaryFile);
assert(isfile(episodeFile), "Missing file: %s", episodeFile);
assert(isfile(pairedFile),  "Missing file: %s", pairedFile);

S = readtable(summaryFile, ...
    "TextType", "string", ...
    "VariableNamingRule", "preserve");

E = readtable(episodeFile, ...
    "TextType", "string", ...
    "VariableNamingRule", "preserve");

P = readtable(pairedFile, ...
    "TextType", "string", ...
    "VariableNamingRule", "preserve");

caseKeys   = ["A_nominal","B_delay","C_target","D_wind","E_mixed"];
caseLabels = ["Nominal","Delay","Target","Wind","Mixed"];

%% ========================================================================
%  FIGURE 1 — Main paper figure
%  PID vs PPO landing success across robustness conditions, with Wilson 95% CI
% =========================================================================
pidRate = nan(1, numel(caseKeys));
ppoRate = nan(1, numel(caseKeys));

pidLow  = nan(size(pidRate));
pidHigh = nan(size(pidRate));
ppoLow  = nan(size(ppoRate));
ppoHigh = nan(size(ppoRate));

for i = 1:numel(caseKeys)
    rPID = S(S.case == caseKeys(i) & S.controller == "PID", :);
    rPPO = S(S.case == caseKeys(i) & S.controller == "PPO", :);

    assert(height(rPID) == 1, "PID summary row missing/duplicated for %s.", caseKeys(i));
    assert(height(rPPO) == 1, "PPO summary row missing/duplicated for %s.", caseKeys(i));

    pidRate(i) = rPID.success_rate_pct;
    ppoRate(i) = rPPO.success_rate_pct;

    pidLow(i)  = rPID.success_ci95_low_pct;
    pidHigh(i) = rPID.success_ci95_high_pct;
    ppoLow(i)  = rPPO.success_ci95_low_pct;
    ppoHigh(i) = rPPO.success_ci95_high_pct;
end

x = 1:numel(caseKeys);
dx = 0.11;

f1 = figure("Color","w", "Units","centimeters", ...
    "Position",[2 2 successFigSizeCm(1) successFigSizeCm(2)]);

hold on;

% Asymmetric Wilson confidence intervals.
hPID = errorbar( ...
    x - dx, pidRate, ...
    pidRate - pidLow, pidHigh - pidRate, ...
    "o", ...
    "LineStyle","none", ...
    "LineWidth", lineWidth, ...
    "MarkerSize", markerSize, ...
    "MarkerFaceColor", "w", ...
    "CapSize", 6, ...
    "DisplayName", "PID only");

hPPO = errorbar( ...
    x + dx, ppoRate, ...
    ppoRate - ppoLow, ppoHigh - ppoRate, ...
    "s", ...
    "LineStyle","none", ...
    "LineWidth", lineWidth, ...
    "MarkerSize", markerSize, ...
    "MarkerFaceColor", "w", ...
    "CapSize", 6, ...
    "DisplayName", "PPO residual + PID");

% Add the PPO-PID change in percentage points.
if showDeltaLabels
    delta = ppoRate - pidRate;

    for i = 1:numel(x)

        if abs(delta(i)) < 0.05
            continue;
        end

        yText = min(99.4, max(pidHigh(i), ppoHigh(i)) + 0.55);

        text(x(i), yText, sprintf("\\Delta %+0.1f pp", delta(i)), ...
            "HorizontalAlignment","center", ...
            "VerticalAlignment","bottom", ...
            "FontName",fontName, ...
            "FontSize",fontSize, ...
            "Interpreter","tex");
    end
end


xlim([0.55 numel(caseKeys)+0.45]);

% A dot/error-bar plot can legitimately use a focused y-range without the
% visual distortion associated with truncated bar charts.
ylim([82 100]);
yticks(82:3:100);

xticks(x);
xticklabels(caseLabels);

ylabel("Landing success rate (%)");

grid on;
box on;
ax = gca;
ax.FontName = fontName;
ax.FontSize = fontSize;
ax.LineWidth = 0.8;
ax.GridAlpha = 0.16;
ax.MinorGridAlpha = 0.08;
ax.TickDir = "out";

legend([hPID hPPO], ...
    "Location","southwest", ...
    "Box","off", ...
    "FontName",fontName, ...
    "FontSize",fontSize);

% Deliberately omit a title for paper use; put the explanation in the caption.

exportgraphics(f1, fullfile(dataDir, "fig_robustness_success.pdf"), ...
    "ContentType","vector");
exportgraphics(f1, fullfile(dataDir, "fig_robustness_success.png"), ...
    "Resolution",600);

%% ========================================================================
%  FIGURE 2 — Detailed Mixed-condition comparison
%  PID / Random / PPO, with episode-derived 95% mean CI
% =========================================================================
controllers = ["PID","Random","PPO"];
controllerLabels = ["PID only","Random residual","PPO residual"];

mixed = E(E.case == "E_mixed", :);

meanXY = nan(1,3);
ciXY   = nan(1,3);

meanVz = nan(1,3);
ciVz   = nan(1,3);

meanVxy = nan(1,3);
ciVxy   = nan(1,3);

for j = 1:numel(controllers)
    T = mixed(mixed.controller == controllers(j), :);
    assert(~isempty(T), "No E_mixed data for controller %s.", controllers(j));

    % Final XY error: all evaluated episodes.
    [meanXY(j), ciXY(j)] = meanCI95(T.xy_error_m);

    % Touchdown metrics: successful landings only.
    Ts = T(T.success == 1, :);
    [meanVz(j),  ciVz(j)]  = meanCI95(Ts.impact_vz_mps);
    [meanVxy(j), ciVxy(j)] = meanCI95(Ts.touchdown_vxy_mps);
end

f2 = figure("Color","w", "Units","centimeters", ...
    "Position",[2 2 mixedFigSizeCm(1) mixedFigSizeCm(2)]);

tl = tiledlayout(1,3, ...
    "TileSpacing","compact", ...
    "Padding","compact");

% ---- (a) Final XY error
nexttile;
plotBarWithCI(meanXY, ciXY, controllerLabels, ...
    "Final XY error (m)", fontName, fontSize);
title("(a)", "FontWeight","normal");

% ---- (b) Vertical touchdown speed
nexttile;
plotBarWithCI(meanVz, ciVz, controllerLabels, ...
    "Touchdown vertical speed (m/s)", fontName, fontSize);
title("(b)", "FontWeight","normal");

% ---- (c) Horizontal touchdown speed
nexttile;
plotBarWithCI(meanVxy, ciVxy, controllerLabels, ...
    "Touchdown horizontal speed (m/s)", fontName, fontSize);
title("(c)", "FontWeight","normal");

exportgraphics(f2, fullfile(dataDir, "fig_mixed_detailed.pdf"), ...
    "ContentType","vector");
exportgraphics(f2, fullfile(dataDir, "fig_mixed_detailed.png"), ...
    "Resolution",600);

%% ========================================================================
%  FIGURE 3 — Optional diagnostic figure
%  Paired PID->PPO rescue / PPO regression counts
% =========================================================================
rescue = zeros(1,numel(caseKeys));
regress = zeros(1,numel(caseKeys));

for i = 1:numel(caseKeys)
    T = P(P.case == caseKeys(i), :);
    rescue(i)  = sum(T.pair_type == "ppo_rescue");
    regress(i) = sum(T.pair_type == "ppo_regression");
end

f3 = figure("Color","w", "Units","centimeters", ...
    "Position",[2 2 pairedFigSizeCm(1) pairedFigSizeCm(2)]);

B = bar(x, [rescue(:), regress(:)], "grouped");
B(1).DisplayName = "PID fail \rightarrow PPO success";
B(2).DisplayName = "PID success \rightarrow PPO fail";

xticks(x);
xticklabels(caseLabels);
ylabel("Number of paired episodes");
ylim([0, max([rescue regress])+1.5]);

grid on;
box on;
ax = gca;
ax.FontName = fontName;
ax.FontSize = fontSize;
ax.LineWidth = 0.8;
ax.GridAlpha = 0.16;
ax.TickDir = "out";

legend("Location","northwest", "Box","off", ...
    "FontName",fontName, "FontSize",fontSize-1);

% Exact McNemar p-values are intentionally not represented as significance
% stars because none of the evaluated cases reached p < 0.05.
for i = 1:numel(caseKeys)
    T = S(S.case == caseKeys(i) & S.controller == "PPO", :);
    if ~isempty(T) && ismember("paired_mcnemar_exact_p", string(T.Properties.VariableNames))
        p = T.paired_mcnemar_exact_p(1);
        y = max(rescue(i), regress(i)) + 0.35;
        text(i, y, sprintf("p = %.3g", p), ...
            "HorizontalAlignment","center", ...
            "FontName",fontName, ...
            "FontSize",fontSize-1);
    end
end

exportgraphics(f3, fullfile(dataDir, "fig_paired_rescue.pdf"), ...
    "ContentType","vector");
exportgraphics(f3, fullfile(dataDir, "fig_paired_rescue.png"), ...
    "Resolution",600);

%% ------------------------------------------------------------------------
%  Console summary
% -------------------------------------------------------------------------
fprintf("\nGenerated paper figures:\n");
fprintf("  %s\n", fullfile(dataDir, "fig_robustness_success.pdf"));
fprintf("  %s\n", fullfile(dataDir, "fig_mixed_detailed.pdf"));
fprintf("  %s\n", fullfile(dataDir, "fig_paired_rescue.pdf"));

fprintf("\nMain success rates [PID, PPO] (%%):\n");
for i = 1:numel(caseKeys)
    fprintf("  %-10s : %5.1f, %5.1f  (delta %+0.1f pp)\n", ...
        caseLabels(i), pidRate(i), ppoRate(i), ppoRate(i)-pidRate(i));
end

%% ========================================================================
%  Local functions
% =========================================================================
function [mu, ciHalf] = meanCI95(x)
    x = x(~isnan(x));
    n = numel(x);

    if n == 0
        mu = NaN;
        ciHalf = NaN;
        return;
    end

    mu = mean(x);

    if n == 1
        ciHalf = NaN;
        return;
    end

    % Student-t 95% CI if Statistics and Machine Learning Toolbox exists.
    % Fall back to normal approximation otherwise.
    s = std(x, 0);
    se = s / sqrt(n);

    if exist("tinv","file") == 2
        tcrit = tinv(0.975, n-1);
    else
        tcrit = 1.96;
    end

    ciHalf = tcrit * se;
end

function plotBarWithCI(mu, ciHalf, labels, yLabelText, fontName, fontSize)
    x = 1:numel(mu);

    b = bar(x, mu, 0.68);
    hold on;

    errorbar(x, mu, ciHalf, ciHalf, ...
        "k", ...
        "LineStyle","none", ...
        "LineWidth",1.0, ...
        "CapSize",5);

    xticks(x);
    xticklabels(labels);
    xtickangle(18);
    ylabel(yLabelText);

    ymax = max(mu + ciHalf, [], "omitnan");
    if isempty(ymax) || isnan(ymax) || ymax <= 0
        ymax = 1;
    end
    ylim([0, ymax*1.18]);

    % Numeric labels help when the differences are small.
    for k = 1:numel(mu)
        text(k, mu(k) + ciHalf(k) + 0.025*ymax, ...
            sprintf("%.4f", mu(k)), ...
            "HorizontalAlignment","center", ...
            "FontName",fontName, ...
            "FontSize",fontSize-1);
    end

    grid on;
    box on;
    ax = gca;
    ax.FontName = fontName;
    ax.FontSize = fontSize;
    ax.LineWidth = 0.8;
    ax.GridAlpha = 0.14;
    ax.TickDir = "out";

    %#ok<NASGU>
    b = b;
end
