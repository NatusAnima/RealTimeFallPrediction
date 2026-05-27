classdef trainer_class
    methods(Static)

        %% ============================================================
        %  TRAINING WRAPPER FOR OFFLINE EEG + ACC PIPELINE
        %% ============================================================

        function trainedClassifier = train_offline_classifier(features, mapped_labels)
            [predictors, predictorNames] = trainer_class.get_predictors(features);
            trainedClassifier = trainer_class.Train_Classifier(features, predictorNames, predictors, mapped_labels);
        end


        %% ============================================================
        %  PREDICTOR TABLE FORMAT
        %% ============================================================

        function [predictors, predictorNames] = get_predictors(features)
            col_length = size(features, 2);
            predictorNames = cell(1, col_length);

            for ii = 1:col_length
                predictorNames{ii} = ['column_', num2str(ii)];
            end

            inputTable = array2table(features, 'VariableNames', predictorNames);
            predictors = inputTable(:, predictorNames);
        end


        %% ============================================================
        %  CLASSIFIER TRAINING
        %  Keeps original lecturer logic:
        %  Bagged ensemble of decision trees with Expected/Unexpected classes
        %% ============================================================

        function trainedClassifier = Train_Classifier(features, predictorNames, predictors, mappedResponse)
            template = templateTree( ...
                'MaxNumSplits', size(features, 1) - 1);

            classNames = {'Expected', 'Unexpected'};

            classificationEnsemble = fitcensemble( ...
                predictors, ...
                mappedResponse, ...
                'Method', 'Bag', ...
                'NumLearningCycles', 30, ...
                'Learners', template, ...
                'ClassNames', classNames);

            predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
            ensemblePredictFcn = @(x) predict(classificationEnsemble, x);

            trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));
            trainedClassifier.ClassificationEnsemble = classificationEnsemble;
            trainedClassifier.PredictorNames = predictorNames;
            trainedClassifier.ClassNames = classNames;
            trainedClassifier.FeatureCount = size(features, 2);
        end


        %% ============================================================
        %  OFFLINE EVALUATION
        %% ============================================================

        function metrics = evaluate_predictions(true_labels, predicted_labels)
            true_binary = trainer_class.labels_to_binary(true_labels);
            pred_binary = trainer_class.labels_to_binary(predicted_labels);

            TP = sum(true_binary == 1 & pred_binary == 1);
            TN = sum(true_binary == 0 & pred_binary == 0);
            FP = sum(true_binary == 0 & pred_binary == 1);
            FN = sum(true_binary == 1 & pred_binary == 0);

            sensitivity = TP / max(TP + FN, eps);
            specificity = TN / max(TN + FP, eps);
            precision = TP / max(TP + FP, eps);
            accuracy = (TP + TN) / max(TP + TN + FP + FN, eps);
            balanced_accuracy = (sensitivity + specificity) / 2;

            metrics = table(TP, TN, FP, FN, sensitivity, specificity, precision, accuracy, balanced_accuracy);
        end


        function binary_labels = labels_to_binary(labels)
            if iscell(labels)
                binary_labels = zeros(numel(labels), 1);
                binary_labels(strcmp(labels, 'Unexpected')) = 1;
            elseif iscategorical(labels)
                labels_cell = cellstr(labels);
                binary_labels = zeros(numel(labels_cell), 1);
                binary_labels(strcmp(labels_cell, 'Unexpected')) = 1;
            else
                binary_labels = labels(:);
                binary_labels(binary_labels == 2) = 0;
            end
        end


        function [validationPredictions, validationScores, validationMetrics] = cross_validate_classifier(trainedClassifier, true_labels, kfold)
            if nargin < 3
                kfold = 10;
            end

            partitionedModel = crossval(trainedClassifier.ClassificationEnsemble, 'KFold', kfold);
            [validationPredictions, validationScores] = kfoldPredict(partitionedModel);

            validationMetrics = trainer_class.evaluate_predictions(true_labels, validationPredictions);
        end


        %% ============================================================
        %  SAVE / LOAD MODEL
        %% ============================================================

        function save_trained_Classifier(path, trainedClassifier)
            classifire = trainedClassifier; %#ok<NASGU>
            save(path, 'classifire');
        end


        function trainedClassifier = load_trained_Classifier(path)
            loaded = load(path);

            if isfield(loaded, 'classifire')
                trainedClassifier = loaded.classifire;
            elseif isfield(loaded, 'trainedClassifier')
                trainedClassifier = loaded.trainedClassifier;
            else
                error('No trained classifier was found in file: %s', path);
            end
        end

    end
end
