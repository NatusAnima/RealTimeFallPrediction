classdef preprocessing
    methods(Static)

        %% ============================================================
        %  OFFLINE DATASET CREATION: EEG + ACC WINDOWS
        %  Keeps original lecturer logic:
        %  EDF -> EEG signal + ACC magnitude -> normalized windows -> labels
        %% ============================================================

        function [eeg_preprocessed_windows, acc_preprocessed_windows, mapped_label_preprocessed_windows, label_preprocessed_windows] = create_eeg_acc_windows( ...
                subjects_list, edfs_name_local, event_table_local, label_table_local, channel, window_size, delay_time, overlap, train_nofall)

            max_windows = length(subjects_list) * length(overlap) * width(event_table_local) * 2;

            eeg_preprocessed_windows = zeros(max_windows, window_size);
            acc_preprocessed_windows = zeros(max_windows, window_size);
            label_preprocessed_windows = zeros(max_windows, 1);

            insert_idx = 1;

            for i = 1:length(subjects_list)
                subject_number = subjects_list(i);
                fprintf('Preprocessing subject %d\n', subject_number);

                [edfs_name, label_table_subject, event_table_subject, eeg_sig, eeg_cap] = ...
                    preprocessing.get_eeg_parameters(subject_number, edfs_name_local, label_table_local, event_table_local, channel);

                acc_sig = preprocessing.get_acc_signal(edfs_name);

                [onset_values, label_values] = preprocessing.get_onset_values( ...
                    length(eeg_sig), event_table_subject, label_table_subject);

                if ~strcmp(train_nofall, 'True')
                    idx_nofall = find(label_values == 2);
                    onset_values(idx_nofall) = [];
                    label_values(idx_nofall) = [];
                end

                [eeg_window, acc_window, label_list] = preprocessing.chope_eeg_acc_2_window( ...
                    eeg_sig, acc_sig, eeg_cap, onset_values, label_values, window_size, delay_time, overlap);

                n = length(label_list);

                eeg_preprocessed_windows(insert_idx:insert_idx+n-1, :) = eeg_window;
                acc_preprocessed_windows(insert_idx:insert_idx+n-1, :) = acc_window;
                label_preprocessed_windows(insert_idx:insert_idx+n-1, :) = label_list;

                insert_idx = insert_idx + n;
            end

            eeg_preprocessed_windows = eeg_preprocessed_windows(1:insert_idx-1, :);
            acc_preprocessed_windows = acc_preprocessed_windows(1:insert_idx-1, :);
            label_preprocessed_windows = label_preprocessed_windows(1:insert_idx-1);

            mapped_label_preprocessed_windows = preprocessing.map_numeric_labels(label_preprocessed_windows);
        end


        %% ============================================================
        %  EDF / SIGNAL LOADING
        %% ============================================================

        function [edfs_name, label_table_subject, event_table_subject, eeg_sig, eeg_cap] = get_eeg_parameters( ...
                subject, edfs_name_local, label_table, event_table, channel)

            if subject < 10
                edf_index = find(contains(edfs_name_local, ['0', num2str(subject)]));
            else
                edf_index = find(contains(edfs_name_local, num2str(subject)));
            end

            if isempty(edf_index)
                error('EDF file for subject %d was not found.', subject);
            end

            edf_index = edf_index(1);
            table_index = find(label_table.Subject == subject);

            if isempty(table_index)
                error('Subject %d was not found in label_table.', subject);
            end

            label_table_subject = label_table{table_index, 3:end};
            event_table_subject = event_table{table_index, 3:end};
            edfs_name = edfs_name_local{edf_index};

            eeg_sig = preprocessing.get_edf_signal(edfs_name, channel);
            eeg_cap = preprocessing.get_eeg_cap(eeg_sig, event_table_subject(1));
        end


        function signal = get_edf_signal(edfs_name, ch)
            EEG_whole = edfread(edfs_name, ...
                'SelectedSignals', ch, ...
                'DataRecordOutputType', 'vector');

            signal = cat(1, EEG_whole.(1){:})';
        end


        function acc_sig = get_acc_signal(edfs_name)
            acc_sig_x = preprocessing.get_edf_signal(edfs_name, 'x_dir') ./ 980;
            acc_sig_y = preprocessing.get_edf_signal(edfs_name, 'y_dir') ./ 980;
            acc_sig_z = preprocessing.get_edf_signal(edfs_name, 'z_dir') ./ 980;

            acc_sig = [acc_sig_x; acc_sig_y; acc_sig_z];
        end


        %% ============================================================
        %  NORMALIZATION
        %% ============================================================

        function eeg_cap = get_eeg_cap(eeg_sig, timing_1)
            per_1 = eeg_sig(timing_1 + 1 : timing_1 + 1000);
            per_1b = per_1 - mean(per_1(1:40));
            eeg_cap = max(per_1b);

            if eeg_cap == 0
                warning('EEG cap is zero. Using eps instead.');
                eeg_cap = eps;
            end
        end


        function eeg_epoch_norm = normolized_eeg(eeg_epoch, eeg_cap)
            eeg_epoch_bn = (eeg_epoch - mean(eeg_epoch(1:40))) ./ eeg_cap;
            eeg_epoch_norm = sgolayfilt(eeg_epoch_bn, 3, 21);
        end


        function acc_epoch_norm = normolized_acc(acc_epoch)
            acc_epoch_x = sgolayfilt(acc_epoch(1, :), 3, 21);
            acc_epoch_y = sgolayfilt(acc_epoch(2, :), 3, 21);
            acc_epoch_z = sgolayfilt(acc_epoch(3, :), 3, 21);

            acc_epoch_norm = sqrt(acc_epoch_x.^2 + acc_epoch_y.^2 + acc_epoch_z.^2);
        end


        %% ============================================================
        %  ONSETS + WINDOWING
        %% ============================================================

        function [onset_values, label_values] = get_onset_values(eeg_length, event_table_subject, label_table_subject)
            onset_nofall = preprocessing.generate_random_numbers( ...
                eeg_length - 2000, event_table_subject, length(event_table_subject));

            onset_values = [onset_nofall, event_table_subject];
            label_values = [2 .* ones(1, length(event_table_subject)), label_table_subject];

            invalid_idx = find(onset_values == -1);
            onset_values(invalid_idx) = [];
            label_values(invalid_idx) = [];

            [onset_values, sort_idx] = sort(onset_values);
            label_values = label_values(sort_idx);
        end


        function random_list = generate_random_numbers(final_number, given_numbers, num_random)
            random_list = [];
            min_step_distance = 2000;

            while length(random_list) < num_random
                rand_num = randi([1000, final_number]);

                valid = true;

                for i = 1:length(given_numbers)
                    if abs(rand_num - given_numbers(i)) < min_step_distance
                        valid = false;
                        break;
                    end
                end

                if valid
                    for i = 1:length(random_list)
                        if abs(rand_num - random_list(i)) < min_step_distance
                            valid = false;
                            break;
                        end
                    end
                end

                if valid
                    random_list = [random_list, rand_num]; %#ok<AGROW>
                end
            end
        end


        function [eeg_window, acc_window, label_list] = chope_eeg_acc_2_window( ...
                eeg_sig, acc_sig, eeg_cap, onset_values, label_values, window_size, delay_time, overlap)

            num_loops = length(onset_values);

            eeg_window = zeros(num_loops * length(overlap), window_size);
            acc_window = zeros(num_loops * length(overlap), window_size);
            label_list = zeros(num_loops * length(overlap), 1);

            row_index = 1;

            for i = 1:num_loops
                for j = 1:length(overlap)
                    start_idx = onset_values(i) + delay_time + overlap(j);
                    end_idx = start_idx + window_size - 1;

                    if start_idx < 1 || end_idx > length(eeg_sig)
                        continue;
                    end

                    eeg_epoch = eeg_sig(start_idx:end_idx);
                    acc_epoch = acc_sig(:, start_idx:end_idx);

                    eeg_window(row_index, :) = preprocessing.normolized_eeg(eeg_epoch, eeg_cap);
                    acc_window(row_index, :) = preprocessing.normolized_acc(acc_epoch);
                    label_list(row_index) = label_values(i);

                    row_index = row_index + 1;
                end
            end

            eeg_window = eeg_window(1:row_index-1, :);
            acc_window = acc_window(1:row_index-1, :);
            label_list = label_list(1:row_index-1);

            [eeg_window, acc_window, label_list] = preprocessing.remove_constant_windows_eeg_acc( ...
                eeg_window, acc_window, label_list);
        end


        function [filtered_eeg, filtered_acc, filtered_labels] = remove_constant_windows_eeg_acc( ...
                eeg_window, acc_window, label_list)

            rows_to_keep = true(size(eeg_window, 1), 1);

            for i = 1:size(eeg_window, 1)
                eeg_is_constant = all(eeg_window(i, :) == eeg_window(i, 1));
                acc_is_constant = all(acc_window(i, :) == acc_window(i, 1));

                if eeg_is_constant || acc_is_constant
                    rows_to_keep(i) = false;
                end
            end

            filtered_eeg = eeg_window(rows_to_keep, :);
            filtered_acc = acc_window(rows_to_keep, :);
            filtered_labels = label_list(rows_to_keep);
        end
        
        function [eeg_windows_L6, eeg_windows_R6, acc_windows, mapped_labels, numeric_labels, subject_per_window] = create_l6_r6_acc_windows( ...
        subjects_list, edfs_name_local, event_table_local, label_table_local, window_size, delay_time, overlap, train_nofall)

            max_windows = length(subjects_list) * length(overlap) * width(event_table_local) * 2;
        
            eeg_windows_L6 = zeros(max_windows, window_size);
            eeg_windows_R6 = zeros(max_windows, window_size);
            acc_windows = zeros(max_windows, window_size);
            numeric_labels = zeros(max_windows, 1);
            subject_per_window = zeros(max_windows, 1);
        
            insert_idx = 1;
        
            for i = 1:length(subjects_list)
        
                subject_number = subjects_list(i);
                fprintf('Preprocessing subject %d: L6 + R6 + ACC\n', subject_number);
        
                [edfs_name, label_table_subject, event_table_subject, eeg_sig_L6, eeg_cap_L6] = ...
                    preprocessing.get_eeg_parameters(subject_number, edfs_name_local, label_table_local, event_table_local, "L6");
        
                [~, ~, ~, eeg_sig_R6, eeg_cap_R6] = ...
                    preprocessing.get_eeg_parameters(subject_number, edfs_name_local, label_table_local, event_table_local, "R6");
        
                acc_sig = preprocessing.get_acc_signal(edfs_name);
        
                [onset_values, label_values] = preprocessing.get_onset_values( ...
                    length(eeg_sig_R6), event_table_subject, label_table_subject);
        
                if ~strcmp(train_nofall, 'True')
                    idx_nofall = find(label_values == 2);
                    onset_values(idx_nofall) = [];
                    label_values(idx_nofall) = [];
                end
        
                for event_idx = 1:length(onset_values)
                    for overlap_idx = 1:length(overlap)
        
                        start_idx = onset_values(event_idx) + delay_time + overlap(overlap_idx);
                        end_idx = start_idx + window_size - 1;
        
                        if start_idx < 1 || end_idx > length(eeg_sig_R6)
                            continue;
                        end
        
                        eeg_epoch_L6 = eeg_sig_L6(start_idx:end_idx);
                        eeg_epoch_R6 = eeg_sig_R6(start_idx:end_idx);
                        acc_epoch = acc_sig(:, start_idx:end_idx);
        
                        eeg_windows_L6(insert_idx, :) = preprocessing.normolized_eeg(eeg_epoch_L6, eeg_cap_L6);
                        eeg_windows_R6(insert_idx, :) = preprocessing.normolized_eeg(eeg_epoch_R6, eeg_cap_R6);
                        acc_windows(insert_idx, :) = preprocessing.normolized_acc(acc_epoch);
        
                        numeric_labels(insert_idx) = label_values(event_idx);
                        subject_per_window(insert_idx) = subject_number;
        
                        insert_idx = insert_idx + 1;
                    end
                end
            end
        
            eeg_windows_L6 = eeg_windows_L6(1:insert_idx-1, :);
            eeg_windows_R6 = eeg_windows_R6(1:insert_idx-1, :);
            acc_windows = acc_windows(1:insert_idx-1, :);
            numeric_labels = numeric_labels(1:insert_idx-1);
            subject_per_window = subject_per_window(1:insert_idx-1);
        
            rows_to_keep = true(size(eeg_windows_R6, 1), 1);
        
            for i = 1:size(eeg_windows_R6, 1)
                if all(eeg_windows_L6(i, :) == eeg_windows_L6(i, 1)) || ...
                   all(eeg_windows_R6(i, :) == eeg_windows_R6(i, 1)) || ...
                   all(acc_windows(i, :) == acc_windows(i, 1))
                    rows_to_keep(i) = false;
                end
            end
        
            eeg_windows_L6 = eeg_windows_L6(rows_to_keep, :);
            eeg_windows_R6 = eeg_windows_R6(rows_to_keep, :);
            acc_windows = acc_windows(rows_to_keep, :);
            numeric_labels = numeric_labels(rows_to_keep);
            subject_per_window = subject_per_window(rows_to_keep);
        
            mapped_labels = preprocessing.map_numeric_labels(numeric_labels);
        end

        %% ============================================================
        %  N4SID FEATURE EXTRACTION
        %% ============================================================

        function my_feature = feature_extraction(Data, sys_order)

            Ts = 1;
            Data_Length = size(Data, 1);
            feature_length = sys_order^2 + sys_order * 2;
        
            my_feature = zeros(Data_Length, feature_length);
        
            h = waitbar(0, 'Starting N4SID feature extraction...');
            tStart = tic;
        
            for j = 1:Data_Length
        
                dy_sys = iddata(Data(j, :)', [], Ts, ...
                    'TimeUnit', 'milliseconds', ...
                    'Tstart', 0);
        
                sys_ss = n4sid(dy_sys, sys_order);
        
                sysA = [];
        
                for k = 1:sys_order
                    sysA = [sysA, sys_ss.A(k, :)];
                end
        
                my_feature(j, :) = [sysA, sys_ss.C, sys_ss.K'];
        
                elapsed = toc(tStart);
                avg_time = elapsed / j;
                remaining = avg_time * (Data_Length - j);
        
                waitbar(j / Data_Length, h, sprintf( ...
                    'N4SID %d/%d | Remaining: %.1f sec', ...
                    j, Data_Length, remaining));
            end
        
            close(h);
        end

        function features = extract_eeg_acc_features(eeg_windows, acc_windows, sys_order)
            eeg_feature = preprocessing.feature_extraction(eeg_windows, sys_order);
            acc_feature = preprocessing.feature_extraction(acc_windows, sys_order);

            features = [eeg_feature, acc_feature];
        end


        function mapped_labels = map_numeric_labels(label_values)
            mapped_labels = cell(size(label_values));

            mapped_labels(label_values == 0) = {'Expected'};
            mapped_labels(label_values == 1) = {'Unexpected'};
            mapped_labels(label_values == 2) = {'Expected'};
        end


        %% ============================================================
        %  THRESHOLD HELPER FOR LATER REAL-TIME USE
        %% ============================================================

        function thresholds = get_signal_threshold(subjects_list, edfs_names, event_table, label_table, channel, window_size, delay_time)
            overlap = 1;
            train_nofall = 'False';

            [eeg_signals, acc_signals, mapped_labels] = preprocessing.create_eeg_acc_windows( ...
                subjects_list, edfs_names, event_table, label_table, channel, window_size + delay_time, 0, overlap, train_nofall);

            labels = zeros(size(eeg_signals, 1), 1);
            labels(strcmp(mapped_labels, 'Unexpected')) = 1;

            eeg_max_values = max(eeg_signals, [], 2);
            eeg_max_unexpected = eeg_max_values(labels == 1);
            eeg_threshold = min(eeg_max_unexpected);

            acc_max_values = max(acc_signals, [], 2);
            acc_max_unexpected = acc_max_values(labels == 1);
            acc_threshold = min(acc_max_unexpected);

            thresholds = [eeg_threshold, acc_threshold];
        end

    end
end
