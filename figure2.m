% Script to visualize RNA prediction metrics from else.txt

% Load data
data = load('0929mamba.txt'); % Format: [length, perplexity, recovery, edit_dist, sc_score]
lengths = data(:, 1);
perplexity = data(:, 2);
recovery = data(:, 3);
edit_dist = data(:, 4);
sc_score = data(:, 5);

% Compute unique lengths and their average metrics
[unique_lengths, ~, idx] = unique(lengths);
n = length(unique_lengths);
avg_perplexity = zeros(n,1);
avg_recovery = zeros(n,1);
avg_edit_dist = zeros(n,1);
avg_sc_score = zeros(n,1);

for i = 1:n
    avg_perplexity(i) = mean(perplexity(idx == i));
    avg_recovery(i) = mean(recovery(idx == i));
    avg_edit_dist(i) = mean(edit_dist(idx == i));
    avg_sc_score(i) = mean(sc_score(idx == i));
end

% Prepare 2x2 subplot
figure('Name','RNA Prediction Metrics','NumberTitle','off');
metrics = {avg_perplexity, avg_recovery, avg_edit_dist, avg_sc_score};
titles = {
    'Variation of Perplexity with RNA Sequence Length', ...
    'Variation of Recovery Rate with RNA Sequence Length', ...
    'Variation of Edit Distance with RNA Sequence Length', ...
    'Variation of Structural Conservation Score with RNA Sequence Length'
};
ylabels = {'Average Perplexity', 'Average Recovery Rate', ...
           'Average Edit Distance', 'Average SC Score'};

% Set Times New Roman font for all text
set(0,'defaultAxesFontName','Times New Roman');
set(0,'defaultTextFontName','Times New Roman');

% 设置统一的散点图样式
markerSize = 10;      % 点的大小
markerColor = [0.9, 0.2, 0.2]; % 红色系
lineWidth = 1;      % 标记边缘线宽

for i = 1:4
    subplot(2,2,i);
    % 绘制散点图（带填充的圆形标记）
    scatter(unique_lengths, metrics{i}, markerSize, ...
            'MarkerEdgeColor', markerColor, ...
            'MarkerFaceColor', markerColor, ...
            'LineWidth', lineWidth);
    
    xlabel('RNA Sequence Length (nt)','FontName','Times New Roman');
    ylabel(ylabels{i},'FontName','Times New Roman');
    title(titles{i},'FontName','Times New Roman');
    grid on;
    set(gca,'FontName','Times New Roman');
end

% Adjust layout
sgtitle('mamba','FontName','Times New Roman');