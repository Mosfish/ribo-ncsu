function data_analysis_comparison()
    % Main function to analyze and compare data from three datasets
    
    fprintf('Starting data analysis...\n');
    
    % Define file names
    GVPTransformer_file = 'ls32.txt';
    my_file = 'ls2e.txt';
    grna_file = 'grna_100.txt';
    
    % Check if files exist
    if ~exist(GVPTransformer_file, 'file')
        fprintf('Error: File %s does not exist in current directory\n', GVPTransformer_file);
        fprintf('Current directory: %s\n', pwd);
        list_txt_files();
        return;
    end
    
    if ~exist(my_file, 'file')
        fprintf('Error: File %s does not exist in current directory\n', my_file);
        fprintf('Current directory: %s\n', pwd);
        list_txt_files();
        return;
    end
    
    if ~exist(grna_file, 'file')
        fprintf('Error: File %s does not exist in current directory\n', grna_file);
        fprintf('Current directory: %s\n', pwd);
        list_txt_files();
        return;
    end
    
    % Read data from all three files
    GVPTransformer_data = read_data_file(GVPTransformer_file);
    my_data = read_data_file(my_file);
    grna_data = read_data_file(grna_file);
    
    if isempty(GVPTransformer_data) || isempty(my_data) || isempty(grna_data)
        fprintf('Failed to read data files. Exiting...\n');
        return;
    end
    
    fprintf('Successfully loaded data:\n');
    fprintf('- %s: %d records\n', GVPTransformer_file, size(GVPTransformer_data, 1));
    fprintf('- %s: %d records\n', my_file, size(my_data, 1));
    fprintf('- %s: %d records\n', grna_file, size(grna_data, 1));
    
    % Analyze data for all three datasets
    results_GVPTransformer = analyze_data_by_length(GVPTransformer_data, 'GVPTransformer');
    results_my = analyze_data_by_length(my_data, 'LongShortGVP');
    results_grna = analyze_data_by_length(grna_data, 'grnade');
    
    % Display analysis results
    display_results(results_GVPTransformer, results_my, results_grna);
    
    % Create comparison plots
    create_comparison_plots(results_GVPTransformer, results_my, results_grna);
    
    fprintf('Analysis completed!\n');
end

function data = read_data_file(filename)
    % Read data file with 5 columns
    % Returns matrix with columns: length, col2, recovery, col4, sc_score
    
    try
        data = load(filename);
        if size(data, 2) < 5
            fprintf('Warning: File %s has fewer than 5 columns\n', filename);
            data = [];
            return;
        end
        fprintf('Successfully read %s with %d rows and %d columns\n', ...
                filename, size(data, 1), size(data, 2));
    catch ME
        fprintf('Error reading file %s: %s\n', filename, ME.message);
        data = [];
    end
end

function results = analyze_data_by_length(data, dataset_name)
    % Analyze data by length ranges and calculate statistics
    
    % Extract columns: length (col 1), recovery (col 3), sc_score (col 5)
    length_col = data(:, 1);
    recovery_col = data(:, 3);
    sc_score_col = data(:, 5);
    
    % Define length ranges: 0-100, 100-200, 200+
    range_0_100 = (length_col >= 0) & (length_col < 100);
    range_100_200 = (length_col >= 100) & (length_col < 200);
    range_200_plus = (length_col >= 200);
    
    % Calculate statistics for range 0-100
    recovery_0_100 = recovery_col(range_0_100);
    sc_score_0_100 = sc_score_col(range_0_100);
    
    if ~isempty(recovery_0_100)
        recovery_mean_0_100 = mean(recovery_0_100);
        recovery_median_0_100 = median(recovery_0_100);
        sc_score_mean_0_100 = mean(sc_score_0_100);
        sc_score_median_0_100 = median(sc_score_0_100);
    else
        recovery_mean_0_100 = 0;
        recovery_median_0_100 = 0;
        sc_score_mean_0_100 = 0;
        sc_score_median_0_100 = 0;
    end
    
    % Calculate statistics for range 100-200
    recovery_100_200 = recovery_col(range_100_200);
    sc_score_100_200 = sc_score_col(range_100_200);
    
    if ~isempty(recovery_100_200)
        recovery_mean_100_200 = mean(recovery_100_200);
        recovery_median_100_200 = median(recovery_100_200);
        sc_score_mean_100_200 = mean(sc_score_100_200);
        sc_score_median_100_200 = median(sc_score_100_200);
    else
        recovery_mean_100_200 = 0;
        recovery_median_100_200 = 0;
        sc_score_mean_100_200 = 0;
        sc_score_median_100_200 = 0;
    end
    
    % Calculate statistics for range 200+
    recovery_200_plus = recovery_col(range_200_plus);
    sc_score_200_plus = sc_score_col(range_200_plus);
    
    if ~isempty(recovery_200_plus)
        recovery_mean_200_plus = mean(recovery_200_plus);
        recovery_median_200_plus = median(recovery_200_plus);
        sc_score_mean_200_plus = mean(sc_score_200_plus);
        sc_score_median_200_plus = median(sc_score_200_plus);
    else
        recovery_mean_200_plus = 0;
        recovery_median_200_plus = 0;
        sc_score_mean_200_plus = 0;
        sc_score_median_200_plus = 0;
    end
    
    % Store results in structure
    results.dataset_name = dataset_name;
    results.count_0_100 = sum(range_0_100);
    results.count_100_200 = sum(range_100_200);
    results.count_200_plus = sum(range_200_plus);
    results.recovery_mean_0_100 = recovery_mean_0_100;
    results.recovery_median_0_100 = recovery_median_0_100;
    results.recovery_mean_100_200 = recovery_mean_100_200;
    results.recovery_median_100_200 = recovery_median_100_200;
    results.recovery_mean_200_plus = recovery_mean_200_plus;
    results.recovery_median_200_plus = recovery_median_200_plus;
    results.sc_score_mean_0_100 = sc_score_mean_0_100;
    results.sc_score_median_0_100 = sc_score_median_0_100;
    results.sc_score_mean_100_200 = sc_score_mean_100_200;
    results.sc_score_median_100_200 = sc_score_median_100_200;
    results.sc_score_mean_200_plus = sc_score_mean_200_plus;
    results.sc_score_median_200_plus = sc_score_median_200_plus;
end

function display_results(results_GVPTransformer, results_my, results_grna)
    % Display analysis results in console for all three datasets
    
    fprintf('\n=== DATA ANALYSIS RESULTS ===\n');
    
    % GVPTransformer results
    fprintf('\n%s:\n', results_GVPTransformer.dataset_name);
    fprintf('Length 0-100 range: %d records\n', results_GVPTransformer.count_0_100);
    fprintf('  Recovery - Mean: %.4f, Median: %.4f\n', ...
            results_GVPTransformer.recovery_mean_0_100, results_GVPTransformer.recovery_median_0_100);
    fprintf('  SC Score - Mean: %.4f, Median: %.4f\n', ...
            results_GVPTransformer.sc_score_mean_0_100, results_GVPTransformer.sc_score_median_0_100);
    
    fprintf('Length 100-200 range: %d records\n', results_GVPTransformer.count_100_200);
    fprintf('  Recovery - Mean: %.4f, Median: %.4f\n', ...
            results_GVPTransformer.recovery_mean_100_200, results_GVPTransformer.recovery_median_100_200);
    fprintf('  SC Score - Mean: %.4f, Median: %.4f\n', ...
            results_GVPTransformer.sc_score_mean_100_200, results_GVPTransformer.sc_score_median_100_200);
    
    fprintf('Length 200+ range: %d records\n', results_GVPTransformer.count_200_plus);
    fprintf('  Recovery - Mean: %.4f, Median: %.4f\n', ...
            results_GVPTransformer.recovery_mean_200_plus, results_GVPTransformer.recovery_median_200_plus);
    fprintf('  SC Score - Mean: %.4f, Median: %.4f\n', ...
            results_GVPTransformer.sc_score_mean_200_plus, results_GVPTransformer.sc_score_median_200_plus);
    
    % LongShortGVP results
    fprintf('\n%s:\n', results_my.dataset_name);
    fprintf('Length 0-100 range: %d records\n', results_my.count_0_100);
    fprintf('  Recovery - Mean: %.4f, Median: %.4f\n', ...
            results_my.recovery_mean_0_100, results_my.recovery_median_0_100);
    fprintf('  SC Score - Mean: %.4f, Median: %.4f\n', ...
            results_my.sc_score_mean_0_100, results_my.sc_score_median_0_100);
    
    fprintf('Length 100-200 range: %d records\n', results_my.count_100_200);
    fprintf('  Recovery - Mean: %.4f, Median: %.4f\n', ...
            results_my.recovery_mean_100_200, results_my.recovery_median_100_200);
    fprintf('  SC Score - Mean: %.4f, Median: %.4f\n', ...
            results_my.sc_score_mean_100_200, results_my.sc_score_median_100_200);
    
    fprintf('Length 200+ range: %d records\n', results_my.count_200_plus);
    fprintf('  Recovery - Mean: %.4f, Median: %.4f\n', ...
            results_my.recovery_mean_200_plus, results_my.recovery_median_200_plus);
    fprintf('  SC Score - Mean: %.4f, Median: %.4f\n', ...
            results_my.sc_score_mean_200_plus, results_my.sc_score_median_200_plus);
    
    % grnade results
    fprintf('\n%s:\n', results_grna.dataset_name);
    fprintf('Length 0-100 range: %d records\n', results_grna.count_0_100);
    fprintf('  Recovery - Mean: %.4f, Median: %.4f\n', ...
            results_grna.recovery_mean_0_100, results_grna.recovery_median_0_100);
    fprintf('  SC Score - Mean: %.4f, Median: %.4f\n', ...
            results_grna.sc_score_mean_0_100, results_grna.sc_score_median_0_100);
    
    fprintf('Length 100-200 range: %d records\n', results_grna.count_100_200);
    fprintf('  Recovery - Mean: %.4f, Median: %.4f\n', ...
            results_grna.recovery_mean_100_200, results_grna.recovery_median_100_200);
    fprintf('  SC Score - Mean: %.4f, Median: %.4f\n', ...
            results_grna.sc_score_mean_100_200, results_grna.sc_score_median_100_200);
    
    fprintf('Length 200+ range: %d records\n', results_grna.count_200_plus);
    fprintf('  Recovery - Mean: %.4f, Median: %.4f\n', ...
            results_grna.recovery_mean_200_plus, results_grna.recovery_median_200_plus);
    fprintf('  SC Score - Mean: %.4f, Median: %.4f\n', ...
            results_grna.sc_score_mean_200_plus, results_grna.sc_score_median_200_plus);
end

function create_comparison_plots(results_GVPTransformer, results_my, results_grna)
    % Create comparison bar charts for three datasets
    
    figure('Position', [100, 100, 1400, 900]);
    
    % Prepare data for plotting
    x = 1:3;  % Three categories: 0-100, 100-200, 200+
    width = 0.25;  % Width for three bars
    
    % Recovery means
    subplot(2, 2, 1);
    GVPTransformer_recovery_means = [results_GVPTransformer.recovery_mean_0_100, ...
                              results_GVPTransformer.recovery_mean_100_200, ...
                              results_GVPTransformer.recovery_mean_200_plus];
    my_recovery_means = [results_my.recovery_mean_0_100, ...
                         results_my.recovery_mean_100_200, ...
                         results_my.recovery_mean_200_plus];
    grna_recovery_means = [results_grna.recovery_mean_0_100, ...
                           results_grna.recovery_mean_100_200, ...
                           results_grna.recovery_mean_200_plus];
    
    b1 = bar(x - width, GVPTransformer_recovery_means, width, 'FaceColor', [0.2, 0.6, 0.8], 'FaceAlpha', 0.8);
    hold on;
    b2 = bar(x, my_recovery_means, width, 'FaceColor', [0.8, 0.4, 0.4], 'FaceAlpha', 0.8);
    b3 = bar(x + width, grna_recovery_means, width, 'FaceColor', [0.4, 0.8, 0.4], 'FaceAlpha', 0.8);
    
    % Add value annotations
    add_value_annotations(b1, GVPTransformer_recovery_means);
    add_value_annotations(b2, my_recovery_means);
    add_value_annotations(b3, grna_recovery_means);
    
    title('Recovery Mean Comparison', 'FontWeight', 'bold', 'FontSize', 12);
    xlabel('Length Range');
    ylabel('Recovery Mean');
    set(gca, 'XTick', x, 'XTickLabel', {'0-100', '100-200', '200+'});
    legend('GVPTransformer', 'LongShortGVP', 'grnade', 'Location', 'best');
    grid on;
    grid minor;
    
    % Recovery medians
    subplot(2, 2, 2);
    GVPTransformer_recovery_medians = [results_GVPTransformer.recovery_median_0_100, ...
                                results_GVPTransformer.recovery_median_100_200, ...
                                results_GVPTransformer.recovery_median_200_plus];
    my_recovery_medians = [results_my.recovery_median_0_100, ...
                           results_my.recovery_median_100_200, ...
                           results_my.recovery_median_200_plus];
    grna_recovery_medians = [results_grna.recovery_median_0_100, ...
                             results_grna.recovery_median_100_200, ...
                             results_grna.recovery_median_200_plus];
    
    b4 = bar(x - width, GVPTransformer_recovery_medians, width, 'FaceColor', [0.6, 0.4, 0.8], 'FaceAlpha', 0.8);
    hold on;
    b5 = bar(x, my_recovery_medians, width, 'FaceColor', [0.9, 0.6, 0.2], 'FaceAlpha', 0.8);
    b6 = bar(x + width, grna_recovery_medians, width, 'FaceColor', [0.8, 0.5, 0.5], 'FaceAlpha', 0.8);
    
    % Add value annotations
    add_value_annotations(b4, GVPTransformer_recovery_medians);
    add_value_annotations(b5, my_recovery_medians);
    add_value_annotations(b6, grna_recovery_medians);
    
    title('Recovery Median Comparison', 'FontWeight', 'bold', 'FontSize', 12);
    xlabel('Length Range');
    ylabel('Recovery Median');
    set(gca, 'XTick', x, 'XTickLabel', {'0-100', '100-200', '200+'});
    legend('GVPTransformer', 'LongShortGVP', 'grnade', 'Location', 'best');
    grid on;
    grid minor;
    
    % SC Score means
    subplot(2, 2, 3);
    GVPTransformer_sc_means = [results_GVPTransformer.sc_score_mean_0_100, ...
                        results_GVPTransformer.sc_score_mean_100_200, ...
                        results_GVPTransformer.sc_score_mean_200_plus];
    my_sc_means = [results_my.sc_score_mean_0_100, ...
                   results_my.sc_score_mean_100_200, ...
                   results_my.sc_score_mean_200_plus];
    grna_sc_means = [results_grna.sc_score_mean_0_100, ...
                     results_grna.sc_score_mean_100_200, ...
                     results_grna.sc_score_mean_200_plus];
    
    b7 = bar(x - width, GVPTransformer_sc_means, width, 'FaceColor', [0.9, 0.8, 0.2], 'FaceAlpha', 0.8);
    hold on;
    b8 = bar(x, my_sc_means, width, 'FaceColor', [0.4, 0.7, 0.9], 'FaceAlpha', 0.8);
    b9 = bar(x + width, grna_sc_means, width, 'FaceColor', [0.9, 0.4, 0.7], 'FaceAlpha', 0.8);
    
    % Add value annotations
    add_value_annotations(b7, GVPTransformer_sc_means);
    add_value_annotations(b8, my_sc_means);
    add_value_annotations(b9, grna_sc_means);
    
    title('SC Score Mean Comparison', 'FontWeight', 'bold', 'FontSize', 12);
    xlabel('Length Range');
    ylabel('SC Score Mean');
    set(gca, 'XTick', x, 'XTickLabel', {'0-100', '100-200', '200+'});
    legend('GVPTransformer', 'LongShortGVP', 'grnade', 'Location', 'best');
    grid on;
    grid minor;
    
    % SC Score medians
    subplot(2, 2, 4);
    GVPTransformer_sc_medians = [results_GVPTransformer.sc_score_median_0_100, ...
                          results_GVPTransformer.sc_score_median_100_200, ...
                          results_GVPTransformer.sc_score_median_200_plus];
    my_sc_medians = [results_my.sc_score_median_0_100, ...
                     results_my.sc_score_median_100_200, ...
                     results_my.sc_score_median_200_plus];
    grna_sc_medians = [results_grna.sc_score_median_0_100, ...
                       results_grna.sc_score_median_100_200, ...
                       results_grna.sc_score_median_200_plus];
    
    b10 = bar(x - width, GVPTransformer_sc_medians, width, 'FaceColor', [0.5, 0.9, 0.7], 'FaceAlpha', 0.8);
    hold on;
    b11 = bar(x, my_sc_medians, width, 'FaceColor', [0.7, 0.5, 0.9], 'FaceAlpha', 0.8);
    b12 = bar(x + width, grna_sc_medians, width, 'FaceColor', [0.9, 0.7, 0.5], 'FaceAlpha', 0.8);
    
    % Add value annotations
    add_value_annotations(b10, GVPTransformer_sc_medians);
    add_value_annotations(b11, my_sc_medians);
    add_value_annotations(b12, grna_sc_medians);
    
    title('SC Score Median Comparison', 'FontWeight', 'bold', 'FontSize', 12);
    xlabel('Length Range');
    ylabel('SC Score Median');
    set(gca, 'XTick', x, 'XTickLabel', {'0-100', '100-200', '200+'});
    legend('GVPTransformer', 'LongShortGVP', 'grnade', 'Location', 'best');
    grid on;
    grid minor;
    
    % Save figure
    saveas(gcf, 'data_comparison_analysis_three_datasets.png');
    fprintf('Chart saved as: data_comparison_analysis_three_datasets.png\n');
end

function add_value_annotations(bar_handle, values)
    % Add value annotations on top of bars
    
    x_data = get(bar_handle, 'XData');
    y_data = get(bar_handle, 'YData');
    
    for i = 1:length(values)
        if values(i) > 0
            text(x_data(i), y_data(i) + y_data(i)*0.02, sprintf('%.4f', values(i)), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                'FontSize', 8, 'FontWeight', 'bold');
        end
    end
end

function list_txt_files()
    % List all .txt files in current directory
    
    txt_files = dir('*.txt');
    fprintf('Available .txt files in current directory:\n');
    if isempty(txt_files)
        fprintf('  No .txt files found\n');
    else
        for i = 1:length(txt_files)
            fprintf('  - %s\n', txt_files(i).name);
        end
    end
end

% Run the main function
data_analysis_comparison();