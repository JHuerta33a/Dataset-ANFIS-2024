% ========================================================================
% DESARROLLO DEL MODELO ANFIS JERÁRQUICO PARA PREDICCIÓN DEL ICA (Índice de Calidad del Aire) MEDIANTE 2 SUBSISTEMAS, 
% APLICANDO TECNICAS DE INTERPOLACIÓN PARA LA CREACIÓN DE MAPAS DE DISTRIBUCIÓN DEL ICA.
% Autor: Juan Manuel Huerta Ordaz
% Versión: 16.5.0
% Descripción: Sistema unificado de predicción del Índice de Calidad del Aire (ICA) usando ANFIS con datos de múltiples estaciones
% ========================================================================

%% ========================  Configuración Inicial y Carga de Datos  ========================  %%

% Limpiar workspace y configurar entorno
clearvars; close all; clc;
rng(42, 'twister'); % Semilla fija para consistencia científica
warning('off', 'MATLAB:table:RowsAddedExistingVars');
        
% Carga y validación de datos
fprintf('1. Cargando datos de las estaciones...\n');
data_files = {
    'ICA_datos_procesados_CAP.xlsx', 'CAP';
    'ICA_datos_procesados_EPG.xlsx', 'EPG';
    'ICA_datos_procesados_FEO.xlsx', 'FEO'
};
        
stations_data = cargar_datos_estacion(data_files);

%% ========================  Preprocesamiento y Unificación  ========================  %%

unified_data = unificar_datos(stations_data);
        
% Eliminar valores físicamente imposibles
unified_data = unified_data(unified_data.ICA_Total >= 0 & unified_data.ICA_Total <= 1000, :);
unified_data = unified_data(unified_data.HR >= 0 & unified_data.HR <= 100, :);
unified_data = unified_data(unified_data.TM >= -20 & unified_data.TM <= 60, :);
fprintf('   -> Datos filtrados por límites físicos (Sanity Check).\n');

% Preparación de Variables
[X_contaminantes, X_meteorologicas, Y_target] = prepare_anfis_variables(unified_data);

%% ========================  Validación Cruzada K-Fold  ========================  %%

% Partición de datos
[X1_train, X1_val, X2_train, X2_val, Y_train, Y_val] = partition_data(X_contaminantes, X_meteorologicas, Y_target, unified_data);

usar_modelo_existente = false;
carpeta_modelos = 'modelos_entrenados';

% Preasignación
fisA_trained = [];
fisB_trained = [];
fisMain_trained = [];
modelo_cargado_path = '';

if usar_modelo_existente && exist(carpeta_modelos,'dir')

    archivos = dir(fullfile(carpeta_modelos, 'ANFIS_Jerarquico_*.mat'));

    if ~isempty(archivos)

        [~, idx] = max([archivos.datenum]);
        modelo_cargado_path = fullfile(archivos(idx).folder, archivos(idx).name);
        fprintf('>> Detectado modelo entrenado: %s\n', modelo_cargado_path);

        try

            S = load(modelo_cargado_path);

            if isfield(S,'fisA_trained') && isfield(S,'fisB_trained') && isfield(S,'fisMain_trained')

                fisA_trained    = S.fisA_trained;
                fisB_trained    = S.fisB_trained;
                fisMain_trained = S.fisMain_trained;

                if isfield(S, 'metricas_kfold')

                    metricas_kfold = S.metricas_kfold;

                    if isfield(metricas_kfold, 'rmse_por_fold'), rmse_total = metricas_kfold.rmse_por_fold; end
                    if isfield(metricas_kfold, 'mae_por_fold'), mae_total = metricas_kfold.mae_por_fold; end
                    if isfield(metricas_kfold, 'r2_por_fold'), r2_total = metricas_kfold.r2_por_fold; end
                    if isfield(metricas_kfold, 'K'), K = metricas_kfold.K; end

                end

                fprintf('>> Modelos FIS cargados correctamente.\n');
                omitir_entrenamiento = true;

            else

                fprintf('>> El archivo no contiene los FIS requeridos.\n');
                omitir_entrenamiento = false;

            end

        catch ME

            fprintf('>> Falló la carga del modelo: %s\n', ME.message);
            omitir_entrenamiento = false;

        end
    else
        fprintf('>> No se encontraron .mat en %s. Se entrenará.\n', carpeta_modelos);
        omitir_entrenamiento = false;
    end
else
    omitir_entrenamiento = false;
end

%% ====== Entrenamiento condicionado ====== %%

if ~exist('omitir_entrenamiento','var') || ~omitir_entrenamiento 
        
    % === Configuración K-Fold ===

    if ~exist('K','var') || isempty(K)
        K = 7;
    end

    unified_data = sortrows(unified_data, 'datetime');
    n = height(unified_data);
    fold_size = floor(n/(K+1));

    % Prealocar vectores para métricas por fold (modelo principal)
    rmse_total = nan(K,1);
    mae_total  = nan(K,1);
    r2_total   = nan(K,1);

    % Escalado fijo para Lat/Lon
    Lat_all = X_meteorologicas(:,5);
    Lon_all = X_meteorologicas(:,6);
    
    lat_min = min(Lat_all,[],'omitnan'); 
    lat_max = max(Lat_all,[],'omitnan');
    lon_min = min(Lon_all,[],'omitnan'); 
    lon_max = max(Lon_all,[],'omitnan');
    
    % Guardas globales
    geo_scaler = struct('lat_min',lat_min,'lat_max',lat_max, 'lon_min',lon_min,'lon_max',lon_max);

    best_rmse_global = inf;
    best_fold_idx = 0;
    
    % Variables para guardar al mejor fold
    best_fisA = [];
    best_fisB = [];
    best_fisMain = [];

    for k = 1:K
        fprintf('\n=== Fold %d de %d ===\n', k, K);
   
        % Separación de datos
        val_start = k*fold_size + 1;
        val_end   = (k+1)*fold_size;
    
        train_idx = false(n,1);
        val_idx   = false(n,1);
    
        train_idx(1:val_start-1) = true;
        val_idx(val_start:val_end) = true;
    
        X1_train = X_contaminantes(train_idx,:);   X1_val   = X_contaminantes(val_idx,:);
        X2_train = X_meteorologicas(train_idx,:);  X2_val   = X_meteorologicas(val_idx,:);
        Y_train  = Y_target(train_idx, :);         Y_val    = Y_target(val_idx, :);

        % Normalización por fold
        mu1  = mean(X1_train, 1, 'omitnan');
        sig1 = std(X1_train, 0, 1, 'omitnan');
        sig1(sig1==0) = 1;
        
        X1_train_n = (X1_train - mu1) ./ sig1;
        X1_val_n   = (X1_val   - mu1) ./ sig1;
        
        muY  = mean(Y_train, 'omitnan');
        sigY = std(Y_train, 0, 'omitnan'); if sigY==0, sigY=1; end
        
        Y_train_n = (Y_train - muY) ./ sigY;
        Y_val_n   = (Y_val   - muY) ./ sigY;

        % Subsistema A —- Contaminantes
        contaminantes_names = {'ICA_CO', 'ICA_SO2', 'ICA_O3', 'ICA_NO2', 'ICA_PM'};
        ica_labels = {'Buena', 'Moderada', 'Insalubre', 'MuyInsalubre', 'Peligrosa'};
        ica_centers = [25, 75, 125, 175, 250];
        ica_sigma = 25;
    
        fisA = sugfis('Name','ContaminantesFIS');
    
        for i = 1:numel(contaminantes_names)
            varname = contaminantes_names{i}; 
            range_i_n = [min(X1_train_n(:,i)) max(X1_train_n(:,i))];
            fisA = addInput(fisA, range_i_n, 'Name', varname);
    
            for j = 1:numel(ica_centers)
                center_n = (ica_centers(j) - mu1(i)) / sig1(i);
                sigma_n  = ica_sigma / sig1(i);
            
                fisA = addMF(fisA, varname, 'gaussmf', [sigma_n center_n], 'Name', ica_labels{j});
            end
        end

        fisA = addOutput(fisA, [-3 3], 'Name','ICA_SubA');
        numInputsA = numel(contaminantes_names);
        ruleListA = [];
        r = 0;

        for i = 1:numInputsA
            for c = 1:5
                r = r + 1;
        
                antecedent = zeros(1, numInputsA);
                antecedent(i) = c;
                ruleListA(r,:) = [antecedent  r  1  1];

            end
        end
        
        % Salida
        fisA.Outputs(1).MembershipFunctions = [];
        nRulesA = size(ruleListA,1);
        coefA = zeros(1, numInputsA + 1);
        
        for kR = 1:nRulesA
            fisA = addMF(fisA, 'ICA_SubA', 'linear', coefA);
            ruleListA(kR, numInputsA+1) = kR;
        end
        
        fisA = addRule(fisA, ruleListA);
        
        % Entrenamiento
        optionsA = anfisOptions('InitialFIS', fisA, 'EpochNumber', 200, 'ValidationData', [X1_val_n Y_val_n], 'DisplayFinalResults', 0);
        
        fisA_trained = anfis([X1_train_n Y_train_n], optionsA);
        
        Y1_train_pred_n = evalfis(fisA_trained, X1_train_n);
        Y1_val_pred_n   = evalfis(fisA_trained, X1_val_n);

        % desnormalizar para el main
        Y1_train_pred = Y1_train_pred_n*sigY + muY;
        Y1_val_pred   = Y1_val_pred_n*sigY + muY;        
        
        train_error_A_real = mean((Y_train - Y1_train_pred).^2);
        
        val_error_A_real   = mean((Y_val   - Y1_val_pred).^2);
        
        % Verificar sobreajuste
        fprintf(' -> Error entrenamiento: %.4f, Error validación: %.4f\n', train_error_A_real, val_error_A_real);

        if val_error_A_real > 1.5 * train_error_A_real
            warning('Posible sobreajuste en Subsistema A (ratio validación/entrenamiento: %.2f)', val_error_A_real/train_error_A_real);
        end
    
        % Subsistema B —- Meteorológicas
        input_names = {'TM','HR','VV','DV_sin','DV_cos','Lat','Lon'};
        
        DV_rad_train = deg2rad(X2_train(:,4));
        DV_sin_train = sin(DV_rad_train);
        DV_cos_train = cos(DV_rad_train);

        DV_rad_val = deg2rad(X2_val(:,4));
        DV_sin_val = sin(DV_rad_val);
        DV_cos_val = cos(DV_rad_val);
    
        % Lat/Lon escalado Global a [0,1]
        Lat_train_g = (X2_train(:,5) - lat_min) ./ (lat_max - lat_min);
        Lon_train_g = (X2_train(:,6) - lon_min) ./ (lon_max - lon_min);
        Lat_val_g   = (X2_val(:,5) - lat_min) ./ (lat_max - lat_min);
        Lon_val_g   = (X2_val(:,6) - lon_min) ./ (lon_max - lon_min);
        
        Lat_train_g = max(0,min(1,Lat_train_g));
        Lon_train_g = max(0,min(1,Lon_train_g));
        Lat_val_g   = max(0,min(1,Lat_val_g));
        Lon_val_g   = max(0,min(1,Lon_val_g));

        X2_train = [X2_train(:,1:3), DV_sin_train, DV_cos_train, Lat_train_g, Lon_train_g];
        X2_val   = [X2_val(:,1:3),   DV_sin_val,   DV_cos_val,   Lat_val_g,   Lon_val_g];

        % Normalización por fold (Z-score)
        X2_train_n = X2_train; 
        X2_val_n   = X2_val;

        mu2  = mean(X2_train(:,1:5), 1, 'omitnan');
        sig2 = std(X2_train(:,1:5), 0, 1, 'omitnan');
        
        sig2(sig2==0) = 1;

        X2_train_n(:,1:5) = (X2_train(:,1:5) - mu2) ./ sig2;
        X2_val_n(:,1:5)   = (X2_val(:,1:5)   - mu2) ./ sig2;

        X2_train_B = fillmissing(X2_train_n,'linear',1,'EndValues','nearest');
        X2_val_B   = fillmissing(X2_val_n,  'linear',1,'EndValues','nearest');
        
        fisB = sugfis('Name','FisB');
        
        for i = 1:numel(input_names)
            varname = input_names{i};
            input_i = X2_train_B(:,i);       
            if strcmp(varname,'Lat') || strcmp(varname,'Lon')
                range_i_n = [0 1];
            else
                vals = input_i(isfinite(input_i));
                if numel(vals) < 10
                    range_i_n = [-3 3];
                else
                    pr = prctile(vals,[1 99]);
                    lo = pr(1); hi = pr(2);
                    if ~isfinite(lo), lo=-3; end
                    if ~isfinite(hi), hi= 3; end
                    if hi<=lo, hi = lo+1e-3; end
                    range_i_n = [lo hi];
                end
            end
            fisB = addInput(fisB, range_i_n, 'Name', varname);
        
            if i<=5
                mu = mu2(i); s = sig2(i);
            else
                mu = 0; s = 1;
            end

            switch varname

                % TM (Temperatura) - 3 gaussmf: Fria, Templada, Caliente
                case 'TM'
                    centers_raw = [10 25 35];     % °C aprox
                    sigma_raw   = 5;              % ancho típico
            
                    centers_n = (centers_raw - mu) ./ s;
                    sigma_n   = max(abs(sigma_raw ./ s), 0.2);
            
                    names = {'Fria','Templada','Caliente'};
                    for jmf = 1:3
                        fisB = addMF(fisB, varname, 'gaussmf', [sigma_n centers_n(jmf)], 'Name', names{jmf});
                    end

                % HR (Humedad relativa) - 3 gaussmf: Baja, Media, Alta
                case 'HR'
                    centers_raw = [20 50 85];         % %HR típicos para baja/media/alta
                    sigma_raw   = [15 10 15];         % ancho distinto por categoría
            
                    centers_n = (centers_raw - mu) ./ s;
                    sigma_n   = max(abs(sigma_raw ./ s), 0.2);
            
                    names = {'Baja','Media','Alta'};
                    for jmf = 1:3
                        fisB = addMF(fisB, varname, 'gaussmf', [sigma_n(jmf) centers_n(jmf)], 'Name', names{jmf});
                    end

                % VV (Velocidad del viento) - 3 gaussmf: Baja, Media, Alta
                case 'VV'
                    centers_raw = [1.5 5 10];         % m/s aprox
                    sigma_raw   = [1.5 2 3];
            
                    centers_n = (centers_raw - mu) ./ s;
                    sigma_n   = max(abs(sigma_raw ./ s), 0.2);
            
                    names = {'Baja','Media','Alta'};
                    for jmf = 1:3
                        fisB = addMF(fisB, varname, 'gaussmf', [sigma_n(jmf) centers_n(jmf)], 'Name', names{jmf});
                    end
            
                % DV_sin / DV_cos - 3 gaussmf: Negativo, Cero, Positivo
                case {'DV_sin','DV_cos'}
                    centers_raw = [-1 0 1];      % dominio natural antes de escalar
                    sigma_raw   = [0.5 0.5 0.5];
            
                    centers_n = (centers_raw - mu) ./ s;
                    sigma_n   = max(abs(sigma_raw ./ s), 0.2);
            
                    names = {'Negativo','Cero','Positivo'};
                    for jmf = 1:3
                        fisB = addMF(fisB, varname, 'gaussmf', [sigma_n(jmf) centers_n(jmf)], 'Name', names{jmf});
                    end
            
                % Lat - 2 gaussmf: Sur, Norte (Lat ya está en [0,1])
                case 'Lat'
                    centers = [0.25 0.75];   % "Sur" vs "Norte"
                    sigma   = 0.2;           % ancho razonable
            
                    fisB = addMF(fisB, varname, 'gaussmf', [sigma centers(1)], 'Name','Sur');
                    fisB = addMF(fisB, varname, 'gaussmf', [sigma centers(2)], 'Name','Norte');
            
                % Lon - 2 gaussmf: Oeste, Este (Lon en [0,1])
                case 'Lon'
                    centers = [0.25 0.75];   % "Oeste" vs "Este"
                    sigma   = 0.2;
            
                    fisB = addMF(fisB, varname, 'gaussmf', [sigma centers(1)], 'Name','Oeste');
                    fisB = addMF(fisB, varname, 'gaussmf', [sigma centers(2)], 'Name','Este');
            end
        end
    
        for ii = 1:numel(fisB.Inputs)
            r = fisB.Inputs(ii).Range;
            if ~all(isfinite(r)) || r(2) <= r(1)
                fisB.Inputs(ii).Range = [-3 3];
            elseif (r(2)-r(1)) < 1e-3
                c = mean(r);
                fisB.Inputs(ii).Range = [c-1e-3, c+1e-3];
            end
        end

        % Limpiar filas no finitas
        mask_trB = all(isfinite(X2_train_B),2) & isfinite(Y_train_n);
        mask_vaB = all(isfinite(X2_val_B),2)   & isfinite(Y_val_n);
        
        X2_train_clean = X2_train_B(mask_trB,:);
        Y_train_clean  = Y_train_n(mask_trB);

        X2_val_clean = X2_val_B(mask_vaB,:);
        Y_val_clean  = Y_val_n(mask_vaB);
        
        % Rango de salida
        ylo = min(Y_train_clean,[],'omitnan');
        yhi = max(Y_train_clean,[],'omitnan');
        
        if ~isfinite(ylo), ylo = -1; end
        if ~isfinite(yhi), yhi =  1; end
        if yhi <= ylo
            epsY = 1e-3;
            ylo = ylo - epsY;
            yhi = yhi + epsY;
        end
        
        fisB = addOutput(fisB, [ylo yhi], 'Name','ICA_SubB');

        % ====== REGLAS EXPERTAS PURAS (9 reglas de dispersión/condición) ======
        % Índices MFs:
        % TM: 1 Fria, 2 Templada, 3 Caliente
        % HR: 1 Baja, 2 Media, 3 Alta
        % VV: 1 Baja, 2 Media, 3 Alta
        % DV_sin: 1 Negativo, 2 Cero, 3 Positivo
        % DV_cos: 1 Negativo, 2 Cero, 3 Positivo
        % Lat: 1 Zona1, 2 Zona2
        % Lon: 1 Zona1, 2 Zona2
        % ---------------------------
        ruleListB = [
            3 1 1 0 0 0 0   1  1   1;   % caliente, HR baja, VV baja -> alto meteo
            3 0 1 0 0 0 0   2  0.9 1;   % caliente y VV baja -> alto
            0 0 3 0 0 0 0   3  1   1;   % VV alta -> bajo
            2 0 2 0 0 0 0   4  0.8 1;   % templada y VV media -> medio
            1 3 0 0 0 0 0   5  0.7 1;   % fría y HR alta -> bajo
        
            % Reglas direccionales ahora sobre DV_sin/DV_cos:
            % Ejemplo: "viento hacia norte" ≈ sin ~ 0, cos > 0
            0 0 1 2 3 0 0   6  0.9 1;   % VV baja y (sin=cero, cos=positivo) -> alto
            0 0 3 2 1 0 0   7  0.9 1;   % VV alta y (sin=cero, cos=negativo) -> bajo
        
            0 0 0 0 0 1 1   8  0.6 1;   % Zona1/Zona1 -> medio
            0 0 0 0 0 2 2   9  0.6 1];  % Zona2/Zona2 -> medio

        % Salidas
        fisB.Outputs(1).MembershipFunctions = [];
        nRulesB = size(ruleListB,1);
        numInputsB = numel(input_names);
        coefB = zeros(1, numInputsB + 1);
        
        for kR = 1:nRulesB
            fisB = addMF(fisB, 'ICA_SubB', 'linear', coefB);
            ruleListB(kR, numInputsB+1) = kR;
        end
    
        fisB = addRule(fisB, ruleListB);
        
        Y_train_n = Y_train_n(:);
        Y_val_n   = Y_val_n(:);
        
        % Entrenamiento
        optionsB = anfisOptions('InitialFIS', fisB, 'EpochNumber', 200, 'ValidationData', [X2_val_clean Y_val_clean], 'DisplayFinalResults', 0);
        optionsB.OptimizationMethod      = 0;       % híbrido (LS + gradiente)

        % Gradiente casi congelado (pero >0 para que no truene):
        optionsB.InitialStepSize         = 1e-4;    % MUY pequeño, pero positivo
        optionsB.ErrorGoal               = 1e-3;    % evita goal=0 ridículo
        
        for j = 1:size(X2_train_clean, 2)
            col = X2_train_clean(:, j);
            finite_col = col(isfinite(col));
        
            if numel(finite_col) <= 1

                % Todo NaN o 1 solo valor => forzar pequeña variación
                pert = linspace(-1, 1, numel(col))';
                pert = pert / max(abs(pert));
                X2_train_clean(:, j) = 1e-3 * pert;
                fprintf(' Columna %d de X2_train_clean sin datos útiles. Se forzó jitter.\n', j);
        
            else
                s = std(finite_col);
                if s < 1e-6

                    % Casi constante => añadir jitter pequeñito
                    pert = linspace(-1, 1, numel(col))';
                    pert = pert / max(abs(pert));
                    X2_train_clean(:, j) = col + 1e-3 * pert;
                    fprintf(' Columna %d de X2_train_clean casi constante (std=%g). Se añadió jitter.\n', j, s);
                end
            end
        end
        
        for j = 1:size(X2_val_clean, 2)
            col = X2_val_clean(:, j);
            finite_col = col(isfinite(col));
            if numel(finite_col) > 1 && std(finite_col) < 1e-6
                pert = linspace(-1, 1, numel(col))';
                pert = pert / max(abs(pert));
                X2_val_clean(:, j) = col + 1e-4 * pert;
            end
        end

        assert(all(isfinite(X2_train_clean),'all'));
        assert(all(isfinite(Y_train_clean),'all'));
        
        test_out = evalfis(fisB, X2_train_clean(1:10,:));
        if any(~isfinite(test_out))
            error('fisB produce NaN/Inf en evalfis antes de entrenar. Revisar MFs.');
        end

        fisB_trained = anfis([X2_train_clean Y_train_clean], optionsB);
        
        for ii = 1:numel(fisB_trained.Inputs)
            r = fisB_trained.Inputs(ii).Range;
            lo = r(1); hi = r(2);
        
            if ~isfinite(lo) || ~isfinite(hi) || hi <= lo
                fprintf(' Corregido rango inválido en %s: [%f %f] --> [-3 3]\n', ...
                    fisB_trained.Inputs(ii).Name, lo, hi);
                fisB_trained.Inputs(ii).Range = [-3 3];     % restaurar rango seguro
            end
        end

        % Predicción normalizada
        Y2_train_pred_n = evalfis(fisB_trained, X2_train_clean);
        Y2_val_pred_n   = evalfis(fisB_trained, X2_val_clean);
    
        % Verificar sobreajuste
        train_error_B = mean((Y_train_n - Y2_train_pred_n).^2);
        val_error_B   = mean((Y_val_n   - Y2_val_pred_n).^2);

        fprintf(' -> Error entrenamiento: %.4f, Error validación: %.4f\n', train_error_B, val_error_B);
        
        if val_error_B > 1.5 * train_error_B
            warning('Posible sobreajuste en Subsistema B (ratio validación/entrenamiento: %.2f)', val_error_B/train_error_B);
        end
    
        % ANFIS Princiapl
        muA  = mean(Y1_train_pred_n);
        sigA = std(Y1_train_pred_n); if sigA==0, sigA=1; end
        
        muB  = mean(Y2_train_pred_n);
        sigB = std(Y2_train_pred_n); if sigB==0, sigB=1; end
        
        SubA_n = (Y1_train_pred_n - muA) ./ sigA;
        SubB_n = (Y2_train_pred_n - muB) ./ sigB;
        
        SubA_val_n = (Y1_val_pred_n - muA) ./ sigA;
        SubB_val_n = (Y2_val_pred_n - muB) ./ sigB;

        X_main_train_n = [SubA_n SubB_n];
        X_main_val_n   = [SubA_val_n SubB_val_n];
    
        % Guardar escalares del Fold
        if k == K
            escala_modelo = struct();
            escala_modelo.mu1  = mu1;
            escala_modelo.sig1 = sig1;
            
            escala_modelo.mu2  = mu2;
            escala_modelo.sig2 = sig2;
            
            escala_modelo.muY  = muY;
            escala_modelo.sigY = sigY;
            
            escala_modelo.muA  = muA;
            escala_modelo.sigA = sigA;
            
            escala_modelo.muB  = muB;
            escala_modelo.sigB = sigB;
        end

        % Estadísticos para definir las gaussmf
        mA = mean(X_main_train_n(:,1), 'omitnan');
        sA = std( X_main_train_n(:,1), 0, 'omitnan');
        if ~isfinite(mA), mA = 0; end
        if ~isfinite(sA) || sA <= 0, sA = 1; end
    
        mB = mean(X_main_train_n(:,2), 'omitnan');
        sB = std( X_main_train_n(:,2), 0, 'omitnan');
        if ~isfinite(mB), mB = 0; end
        if ~isfinite(sB) || sB <= 0, sB = 1; end
    
        % Creamos el FIS principal
        fisMain = sugfis('Name','FisMain');

        % Rango de las entradas basado en datos del train
        rangoA = [min(X_main_train_n(:,1)) max(X_main_train_n(:,1))];
        rangoB = [min(X_main_train_n(:,2)) max(X_main_train_n(:,2))];
    
        if ~all(isfinite(rangoA)) || rangoA(2) <= rangoA(1)
            rangoA = [mA - 3*sA, mA + 3*sA];
        end
        if ~all(isfinite(rangoB)) || rangoB(2) <= rangoB(1)
            rangoB = [mB - 3*sB, mB + 3*sB];
        end
    
        fisMain = addInput(fisMain, rangoA, 'Name','SubA_n');
        fisMain = addInput(fisMain, rangoB, 'Name','SubB_n');
    
        % ====== MFs GAUSSIANAS (Bajo, Medio, Alto) PARA CADA INPUT ======
        % SubA_n
        fisMain = addMF(fisMain, 'SubA_n','gaussmf',[sA mA - sA], 'Name','Bajo');
        fisMain = addMF(fisMain, 'SubA_n','gaussmf',[sA mA      ], 'Name','Medio');
        fisMain = addMF(fisMain, 'SubA_n','gaussmf',[sA mA + sA ], 'Name','Alto');
    
        % SubB_n
        fisMain = addMF(fisMain, 'SubB_n','gaussmf',[sB mB - sB], 'Name','Bajo');
        fisMain = addMF(fisMain, 'SubB_n','gaussmf',[sB mB     ], 'Name','Medio');
        fisMain = addMF(fisMain, 'SubB_n','gaussmf',[sB mB + sB], 'Name','Alto');
    
        % Salida
        ylo_main = min(Y_train_n,[],'omitnan');
        yhi_main = max(Y_train_n,[],'omitnan');
        if ~isfinite(ylo_main), ylo_main = -1; end
        if ~isfinite(yhi_main), yhi_main =  1; end
        if yhi_main <= ylo_main
            yhi_main = ylo_main + 1e-3;
        end
    
        fisMain = addOutput(fisMain, [ylo_main yhi_main], 'Name','ICA_Final_n');

        % ====== REGLAS EXPERTAS PURAS (9 reglas tipo "máximo difuso") ======
        % Salida_Contaminantes: 1 Bajo,2 Medio,3 Alto
        % Salida_Meteorologicas: 1 Bajo,2 Medio,3 Alto
        
        ruleListMain = [
            1 1   1 1 1;
            2 1   2 1 1;
            1 2   3 1 1;
            2 2   4 1 1;
            3 1   5 1 1;
            1 3   6 1 1;
            3 2   7 1 1;
            2 3   8 1 1;
            3 3   9 1 1 ];
        
        fisMain.Outputs(1).MembershipFunctions = [];
        coefMain = zeros(1, 3);
    
        for kR = 1:size(ruleListMain,1)
            fisMain = addMF(fisMain, 'ICA_Final_n', 'linear', coefMain);
            ruleListMain(kR, 3) = kR;
        end
    
        fisMain = addRule(fisMain, ruleListMain);

        % Entrenamiento
        optionsMain = anfisOptions('InitialFIS', fisMain, 'EpochNumber', 200, 'ValidationData', [X_main_val_n Y_val_n], 'DisplayFinalResults', 0);
        
        fisMain_trained = anfis([X_main_train_n Y_train_n], optionsMain);

        for ii = 1:numel(fisMain_trained.Inputs)
            r = fisMain_trained.Inputs(ii).Range;
        
            if ~all(isfinite(r)) || r(2) <= r(1)

                % Caso 1: NaN, Inf o rango invertido → set default
                fisMain_trained.Inputs(ii).Range = [-3 3];
            elseif (r(2) - r(1)) < 1e-3

                % Caso 2: Rango demasiado pequeño → expandir minimamente
                c = mean(r);
                fisMain_trained.Inputs(ii).Range = [c - 1e-3, c + 1e-3];
            end
        end
        
        % Asegurar rangos válidos también para la salida
        r = fisMain_trained.Outputs(1).Range;
        if ~all(isfinite(r)) || r(2) <= r(1)
            fisMain_trained.Outputs(1).Range = [-1 1];
        elseif (r(2)-r(1)) < 1e-3
            c = mean(r);
            fisMain_trained.Outputs(1).Range = [c-1e-3 c+1e-3];
        end

        % Predicción y desnormalización
        Y_main_pred_n = evalfis(fisMain_trained, X_main_val_n);
        Y_main_pred   = Y_main_pred_n*sigY + muY;

        % Visualización de las MFs
        fprintf('Visualizando funciones de membresía del fold %d...\n', k);
        
        % Visualizar MFs de Subsistema A
        figure('Name', sprintf('MFs - Subsistema A (Fold %d)', k));
        for i = 1:numel(contaminantes_names)
            subplot(2,3,i);
            plotmf(fisA_trained, 'input', i);
            title(contaminantes_names{i});
            grid on;
        end
        
        % Visualizar MFs de Subsistema B
        figure('Name', sprintf('MFs - Subsistema B (Fold %d)', k));
        for i = 1:numel(input_names)
            subplot(3,3,i);
            plotmf(fisB_trained, 'input', i);
            title(input_names{i});
            grid on;
        end
        
        % Visualizar MFs del ANFIS principal
        figure('Name', sprintf('MFs - ANFIS Principal (Fold %d)', k));
        for i = 1:2
            subplot(1,2,i);
            plotmf(fisMain_trained, 'input', i);
            title(fisMain_trained.Inputs(i).Name);
            grid on;
        end
    
        % Evaluación del Fold
        Y_pred_n = evalfis(fisMain_trained, X_main_val_n);
        Y_pred   = Y_pred_n*sigY + muY;
        rmse_total(k) = sqrt(mean((Y_val - Y_pred).^2));
        mae_total(k)  = mean(abs(Y_val - Y_pred));
        if std(Y_val)>0 && std(Y_pred)>0
            r2_total(k) = corr(Y_val, Y_pred)^2;
        else
            r2_total(k) = NaN;
        end

        fprintf('Fold %d - RMSE: %.4f | R²: %.4f | MAE: %.4f\n', k, rmse_total(k), r2_total(k), mae_total(k));

        if rmse_total(k) < best_rmse_global
            fprintf(' Nuevo mejor modelo encontrado (Mejora RMSE de %.4f a %.4f)\n', best_rmse_global, rmse_total(k));
            
            best_rmse_global = rmse_total(k);
            best_fold_idx = k;
            
            % Hacemos una copia de seguridad de los objetos FIS en memoria
            best_fisA = fisA_trained;
            best_fisB = fisB_trained;
            best_fisMain = fisMain_trained;
        end

        % Al final del fold k (después de calcular rmse_total(k), mae_total(k), r2_total(k))
        chk_dir = 'modelos_checkpoints';
        if ~exist(chk_dir,'dir'), mkdir(chk_dir); end
        chk_path = fullfile(chk_dir, sprintf('ANFIS_chk_fold%02d.mat', k));
        metricas_fold = struct('k',k, 'RMSE',rmse_total(k), 'MAE',mae_total(k), 'R2',r2_total(k));
        save(chk_path, 'fisA_trained','fisB_trained','fisMain_trained','metricas_fold');
    end

    %% ====== Consolidación de métricas K-Fold ====== %%
    fprintf('\n>>> SELECCIÓN FINAL: El mejor modelo fue el del Fold %d (RMSE=%.4f)\n', best_fold_idx, best_rmse_global);
    fprintf('>>> Restaurando pesos del mejor modelo para la validación final...\n');
    
    fisA_trained = best_fisA;
    fisB_trained = best_fisB;
    fisMain_trained = best_fisMain;
    
    metricas_kfold = struct();
    metricas_kfold.K = K;
    
    % Modelo principal
    metricas_kfold.rmse_por_fold = rmse_total(:);
    metricas_kfold.rmse_prom = mean(rmse_total,'omitnan');
    metricas_kfold.rmse_std  = std(rmse_total,'omitnan');
    
    metricas_kfold.mae_por_fold  = mae_total(:);
    metricas_kfold.mae_prom = mean(mae_total,'omitnan');
    metricas_kfold.mae_std  = std(mae_total,'omitnan');
    
    metricas_kfold.r2_por_fold   = r2_total(:);
    metricas_kfold.r2_prom = mean(r2_total,'omitnan');
    metricas_kfold.r2_std  = std(r2_total,'omitnan');
else
    fprintf('>> Saltando entrenamiento; se usará el modelo cargado: %s\n', modelo_cargado_path);
end

% Verificación de disponibilidad de modelos antes de validar/interpolar
if isempty(fisA_trained) || isempty(fisB_trained) || isempty(fisMain_trained)
    error('Los modelos FIS no están disponibles. Entrene o cargue primero.');
end

%% ====== Interpolación Espacial ====== %%

% --- Coordenadas de las estaciones (Latitud, Longitud) ---
coords_estaciones = [
    20.5986, -100.3800; % CAP
    20.6333, -100.4000; % EPG
    20.6167, -100.4433; % FEO
];
n_estaciones = size(coords_estaciones, 1);

fprintf('Realizando predicción sobre todos los datos para los mapas...\n');

% Regenerar matrices de entrada completas
[X_contaminantes, X_meteorologicas, ~] = prepare_anfis_variables(unified_data);

% Normalizar contaminantes con los escalares del modelo
X1_all_n = (X_contaminantes - escala_modelo.mu1) ./ escala_modelo.sig1;

% Evaluar fisA en espacio normalizado (salida también normalizada)
Y_A_all_n = evalfis(fisA_trained, X1_all_n);
fprintf('Salida de fisA_trained (normalizada): %dx%d\n', size(Y_A_all_n));

% Convertir a SubA_n (la entrada que espera el ANFIS principal)
SubA_all_n = (Y_A_all_n - escala_modelo.muA) ./ escala_modelo.sigA;

% DV en grados -> radianes
DV_rad_all = deg2rad(X_meteorologicas(:,4));
DV_sin_all = sin(DV_rad_all);
DV_cos_all = cos(DV_rad_all);

% Escalado geográfico de Lat/Lon usando los min/max globales
Lat_all = X_meteorologicas(:,5);
Lon_all = X_meteorologicas(:,6);

lat_min = geo_scaler.lat_min;
lat_max = geo_scaler.lat_max;
lon_min = geo_scaler.lon_min;
lon_max = geo_scaler.lon_max;

Lat_all_g = (Lat_all - lat_min) ./ (lat_max - lat_min);
Lon_all_g = (Lon_all - lon_min) ./ (lon_max - lon_min);

Lat_all_g = max(0, min(1, Lat_all_g));
Lon_all_g = max(0, min(1, Lon_all_g));

% Construir la matriz de entrada para SubB (7 columnas)
X2_all_B = [ ...
    X_meteorologicas(:,1:3), ...    % TM, HR, VV
    DV_sin_all, DV_cos_all, ...     % componentes trigonométricas
    Lat_all_g, Lon_all_g];          % Lat/Lon escaladas

% Normalizar SOLO las primeras 5 variables (TM, HR, VV, DV_sin, DV_cos)
X2_all_B(:,1:5) = (X2_all_B(:,1:5) - escala_modelo.mu2) ./ escala_modelo.sig2;

fprintf('X2_all_B (normalizado): %dx%d (esperado: %dx7)\n', ...
    size(X2_all_B,1), size(X2_all_B,2), size(X_meteorologicas,1));

% Evaluar fisB (salida normalizada)
Y_B_all_n = evalfis(fisB_trained, X2_all_B);
fprintf('Salida de fisB_trained (normalizada): %dx%d\n', size(Y_B_all_n));

% Convertir a SubB_n (entrada del main)
SubB_all_n = (Y_B_all_n - escala_modelo.muB) ./ escala_modelo.sigB;

% Entradas que realmente espera el main: [SubA_n, SubB_n]
X_main_all_n = [SubA_all_n, SubB_all_n];

% Salida normalizada del main
Y_final_all_n = evalfis(fisMain_trained, X_main_all_n);
fprintf('      Salida final (normalizada) fisMain_trained: %dx%d\n', size(Y_final_all_n));

% Desnormalizar a escala ICA real
Y_final_all = Y_final_all_n * escala_modelo.sigY + escala_modelo.muY;
fprintf('      Salida final (ICA real): %dx%d\n', size(Y_final_all));

% Extrer ICA predicho para cada estación
ica_observado = zeros(n_estaciones, 1);
ids_estaciones = unified_data.StationID;

for i = 1:n_estaciones
    last_idx = find(ids_estaciones == i, 1, 'last');
    if ~isempty(last_idx)
        ica_observado(i) = Y_final_all(last_idx);
    else
        ica_observado(i) = NaN;
    end
end

fprintf('Valores de ICA a interpolar (modelo ANFIS): CAP=%.2f, EPG=%.2f, FEO=%.2f\n', ...
    ica_observado(1), ica_observado(2), ica_observado(3));

%% ====== Creación de la malla de interpolación para la ZMQ ====== %%

% Se define un área geográfica (bounding box) para la ZMQ
lat_lim = [20.55, 20.68]; % Límites de latitud
lon_lim = [-100.50, -100.35]; % Límites de longitud

% Se crea una malla con una resolución de 100x100 puntos
[lon_grid, lat_grid] = meshgrid(linspace(lon_lim(1), lon_lim(2), 100), ...
                                linspace(lat_lim(1), lat_lim(2), 100));

% Implementación de las técnicas de interpolación

% RGB según la tabla de la norma
colores_ica = [
    0,   228, 0;    % Verde (Buena) 
    255, 255, 0;    % Amarillo (Aceptable)
    255, 126, 0;    % Naranja (Mala)
    255, 0,   0;    % Rojo (Muy Mala)
    143, 63,  151   % Morado (Extremadamente Mala)
] / 255; % Normalizar a 0-1

% Definir los rangos de ICA según la norma (asumiendo valores generales)
rangos_ica = [0, 50, 100, 150, 200, 500];

% Función para asignar colores según valor de ICA
function color = asignar_color_ica(valor_ica)
    persistent colores_ica rangos_ica
    if isempty(colores_ica)
        colores_ica = [
            0,   228, 0;    % Verde
            255, 255, 0;    % Amarillo
            255, 126, 0;    % Naranja
            255, 0,   0;    % Rojo
            143, 63,  151   % Morado
        ] / 255;
        rangos_ica = [0, 50, 100, 150, 200, 500];
    end
    
    if valor_ica <= rangos_ica(2)
        color = colores_ica(1,:); % Verde
    elseif valor_ica <= rangos_ica(3)
        color = colores_ica(2,:); % Amarillo
    elseif valor_ica <= rangos_ica(4)
        color = colores_ica(3,:); % Naranja
    elseif valor_ica <= rangos_ica(5)
        color = colores_ica(4,:); % Rojo
    else
        color = colores_ica(5,:); % Morado
    end
end

% IDW (Inverse Distance Weighting)
fprintf('Calculando interpolación IDW...\n');
idw_grid = idw_interpolation(coords_estaciones(:,2), coords_estaciones(:,1), ica_observado, lon_grid, lat_grid);

% Kriging (Simple y Universal)
fprintf('Aplicando Kriging Universal con validación de parámetros...\n');
% Calcular variograma con manejo de nugget
[d, V] = variogram(coords_estaciones, ica_observado, 'plot', false);
% Ajustar modelo con nugget automático
[a, c, n, vstruct] = variogramfit(d, V, [], [], []);
fprintf('Parámetros del variograma: Rango=%.3f, Sill=%.3f, Nugget=%.3f\n', a, c, n);
% Kriging Simple con media estimada
[sk_grid, sk_variance] = kriging(vstruct, coords_estaciones(:,2), ...
    coords_estaciones(:,1), ica_observado, lon_grid, lat_grid, 'method', 'simple');

% Kriging Universal con tendencia lineal
[uk_grid, uk_variance] = kriging(vstruct, coords_estaciones(:,2), ...
    coords_estaciones(:,1), ica_observado, lon_grid, lat_grid, ...
    'method', 'universal', 'trend', 'linear');

fprintf('Interpolación Kriging completada exitosamente\n');

%% === Evaluación y Visualización de Métodos de Interpolación === %%

% Inicializar estructuras para almacenar métricas detalladas
metodos = {'IDW', 'KrigingSimple', 'KrigingUniversal'};
metricas = struct();

% Definir número de estaciones a partir de las coordenadas
nestaciones = size(coords_estaciones, 1);

icaobservado = zeros(nestaciones, 1); % Crea espacio para los ICA

idsestaciones = unified_data.StationID; % Vector de identificadores por muestras

for i = 1:nestaciones
    lastidx = find(idsestaciones == i, 1, 'last');
    if isempty(lastidx)
        icaobservado(i) = NaN;
    else
        icaobservado(i) = unified_data.ICA_Total(lastidx);
    end
end

% Inicializar matrices para métricas por método
for i = 1:length(metodos)
    metricas.(metodos{i}).errores_absolutos = zeros(nestaciones, 1);
    metricas.(metodos{i}).errores_cuadraticos = zeros(nestaciones, 1);
    metricas.(metodos{i}).errores_relativos = zeros(nestaciones, 1);
    metricas.(metodos{i}).predicciones = zeros(nestaciones, 1);
end

%% ======= Validación Leave-One-Out ========= %%

for i = 1:nestaciones
    fprintf('Validando estación %d/%d\n', i, nestaciones);
    
    % Datos de entrenamiento (excluir estación i)
    coords_train = coords_estaciones([1:i-1, i+1:end], :);
    ica_train = icaobservado([1:i-1, i+1:end]);
    
    % Punto de prueba
    coord_test = coords_estaciones(i, :);
    ica_real = icaobservado(i);
    
    try
        % IDW
        pred_idw = idw_interpolation(coords_train(:,1), coords_train(:,2), ica_train, coord_test(1), coord_test(2), 'power', 2);
        metricas.IDW.predicciones(i) = pred_idw;
        
        % Kriging
        if length(ica_train) >= 2
            [d_cv, V_cv] = variogram(coords_train, ica_train, 'plot', false);
            [~, ~, ~, vstruct_cv] = variogramfit(d_cv, V_cv, mean(d_cv), var(V_cv), 0, 'gauss2', false, false);
            
            % Kriging Simple
            pred_sk = kriging(vstruct_cv, coords_train(:,1), coords_train(:,2), ica_train, coord_test(1), coord_test(2), 'method', 'simple');
            metricas.KrigingSimple.predicciones(i) = pred_sk;
            
            % Kriging Universal
            pred_uk = kriging(vstruct_cv, coords_train(:,1), ...
                                      coords_train(:,2), ica_train, coord_test(1), coord_test(2), 'method', 'universal', 'trend', 'linear');
            metricas.KrigingUniversal.predicciones(i) = pred_uk;
        else
            % Usar IDW como fallback
            metricas.KrigingSimple.predicciones(i) = pred_idw;
            metricas.KrigingUniversal.predicciones(i) = pred_idw;
        end
        
        % Calcular métricas para cada método
        for j = 1:length(metodos)
            metodo = metodos{j};
            pred = metricas.(metodo).predicciones(i);
            
            % Error absoluto
            metricas.(metodo).errores_absolutos(i) = abs(ica_real - pred);
            
            % Error cuadrático
            metricas.(metodo).errores_cuadraticos(i) = (ica_real - pred)^2;
            
            % Error relativo (%)
            if ica_real ~= 0
                metricas.(metodo).errores_relativos(i) = abs(ica_real - pred) / abs(ica_real) * 100;
            else
                metricas.(metodo).errores_relativos(i) = 0;
            end
        end
        
    catch ME
        fprintf('Error en estación %d: %s\n', i, ME.message);
        
        % Asignar valores promedio como fallback
        mean_ica = mean(ica_train);
        for j = 1:length(metodos)
            metodo = metodos{j};
            metricas.(metodo).predicciones(i) = mean_ica;
            metricas.(metodo).errores_absolutos(i) = abs(ica_real - mean_ica);
            metricas.(metodo).errores_cuadraticos(i) = (ica_real - mean_ica)^2;
            if ica_real ~= 0
                metricas.(metodo).errores_relativos(i) = abs(ica_real - mean_ica) / abs(ica_real) * 100;
            else
                metricas.(metodo).errores_relativos(i) = 0;
            end
        end
    end
end

%% ======= Calcular métricas estadísticas finales ======%%

resultados_metricas = struct();
for i = 1:length(metodos)
    metodo = metodos{i};
    
    % Métricas principales
    resultados_metricas.(metodo).RMSE = sqrt(mean(metricas.(metodo).errores_cuadraticos));
    resultados_metricas.(metodo).MAE = mean(metricas.(metodo).errores_absolutos);
    resultados_metricas.(metodo).MAPE = mean(metricas.(metodo).errores_relativos);
    
    % Coeficiente de correlación
    if std(icaobservado) > 0 && std(metricas.(metodo).predicciones) > 0
        resultados_metricas.(metodo).R = corr(icaobservado, metricas.(metodo).predicciones);
        resultados_metricas.(metodo).R2 = resultados_metricas.(metodo).R^2;
    else
        resultados_metricas.(metodo).R = NaN;
        resultados_metricas.(metodo).R2 = NaN;
    end
    
    % Estadísticas adicionales
    resultados_metricas.(metodo).MAX_ERROR = max(metricas.(metodo).errores_absolutos);
    resultados_metricas.(metodo).MIN_ERROR = min(metricas.(metodo).errores_absolutos);
    resultados_metricas.(metodo).STD_ERROR = std(metricas.(metodo).errores_absolutos);
end

%% ====== Visualizació =====%%%
% Tabla de Comparación de Métricas
fprintf('\n=== Tabla de Comparación de Métricas ===\n');
fprintf('%-15s | %-8s | %-8s | %-8s | %-8s | %-8s\n', 'Método', 'RMSE', 'MAE', 'MAPE(%)', 'R²', 'Max Error');
fprintf('%s\n', repmat('-', 1, 70));

for i = 1:length(metodos)
    metodo = metodos{i};
    fprintf('%-15s | %8.3f | %8.3f | %8.2f | %8.3f | %8.3f\n', ...
            metodo, ...
            resultados_metricas.(metodo).RMSE, ...
            resultados_metricas.(metodo).MAE, ...
            resultados_metricas.(metodo).MAPE, ...
            resultados_metricas.(metodo).R2, ...
            resultados_metricas.(metodo).MAX_ERROR);
end

% Gráficos de Comparación
figure('Name', 'Comparación de Métricas de Interpolación', 'Position', [100, 100, 1200, 800]);

% Subplot 1: Gráfico de barras con métricas principales
subplot(2, 3, 1);
metricas_plot = [
    cellfun(@(m) resultados_metricas.(m).RMSE, metodos);
    cellfun(@(m) resultados_metricas.(m).MAE, metodos);
    cellfun(@(m) resultados_metricas.(m).MAPE, metodos)
];

bar(metricas_plot');
set(gca, 'XTickLabel', metodos);
xlabel('Método de Interpolación');
ylabel('Valor de la Métrica');
title('Comparación de Métricas de Error');
legend({'RMSE', 'MAE', 'MAPE(%)'}, 'Location', 'best');
grid on;

% Subplot 2: R² por método
subplot(2, 3, 2);
r2_values = cellfun(@(m) resultados_metricas.(m).R2, metodos);
bar(r2_values, 'FaceColor', [0.2, 0.6, 0.8]);
set(gca, 'XTickLabel', metodos);
xlabel('Método de Interpolación');
ylabel('R² (Coeficiente de Determinación)');
title('Calidad de Ajuste por Método');
ylim([0, 1]);
grid on;

% Subplot 3: Box plot de errores absolutos
subplot(2, 3, 3);
error_data = cell(1, length(metodos));
for i = 1:length(metodos)
    error_data{i} = metricas.(metodos{i}).errores_absolutos;
end
boxplot(cell2mat(error_data), repelem(1:length(metodos), nestaciones));
set(gca, 'XTickLabel', metodos);
xlabel('Método de Interpolación');
ylabel('Error Absoluto');
title('Distribución de Errores Absolutos');
grid on;

% Subplot 4: Dispersión Valores Reales vs Predichos
subplot(2, 3, [4, 5, 6]);
colores = {'r', 'g', 'b'};
simbolos = {'o', 's', '^'};

hold on;
for i = 1:length(metodos)
    metodo = metodos{i};
    scatter(icaobservado, metricas.(metodo).predicciones, 100, colores{i}, simbolos{i}, ...
            'DisplayName', sprintf('%s (R²=%.3f)', metodo, resultados_metricas.(metodo).R2));
end

% Línea de referencia perfecta
plot([min(icaobservado), max(icaobservado)], [min(icaobservado), max(icaobservado)], ...
     'k--', 'LineWidth', 2, 'DisplayName', 'Predicción Perfecta');

xlabel('ICA Observado');
ylabel('ICA Predicho');
title('Valores Reales vs Predichos por Método de Interpolación');
legend('Location', 'best');
grid on;

% Análisis de Residuos
figure('Name', 'Análisis de Residuos por Método', 'Position', [200, 50, 1000, 600]);

for i = 1:length(metodos)
    metodo = metodos{i};
    residuos = icaobservado - metricas.(metodo).predicciones;
    
    % Histograma de residuos
    subplot(2, length(metodos), i);
    histogram(residuos, 'Normalization', 'probability', 'FaceColor', colores{i});
    xlabel('Residuo');
    ylabel('Probabilidad');
    title(sprintf('Residuos - %s', metodo));
    grid on;
    
    % Q-Q plot para normalidad
    subplot(2, length(metodos), i + length(metodos));
    qqplot(residuos);
    title(sprintf('Q-Q Plot - %s', metodo));
    grid on;
end

% Mapa de Errores por Estación
figure('Name', 'Errores por Estación y Método', 'Position', [300, 100, 800, 600]);

estaciones_nombres = {'CAP', 'EPG', 'FEO'};
errores_por_estacion = zeros(nestaciones, length(metodos));

for i = 1:length(metodos)
    errores_por_estacion(:, i) = metricas.(metodos{i}).errores_absolutos;
end

bar(errores_por_estacion);
set(gca, 'XTickLabel', estaciones_nombres);
xlabel('Estación');
ylabel('Error Absoluto');
title('Error Absoluto por Estación y Método');
legend(metodos, 'Location', 'best');
grid on;

%% ======== Identificar el mejor método ============ %%
[~, mejor_rmse] = min(cellfun(@(m) resultados_metricas.(m).RMSE, metodos));
[~, mejor_r2] = max(cellfun(@(m) resultados_metricas.(m).R2, metodos));

fprintf('\n=== RECOMENDACIONES ===\n');
fprintf('Mejor método según RMSE: %s (RMSE = %.3f)\n', metodos{mejor_rmse}, ...
        resultados_metricas.(metodos{mejor_rmse}).RMSE);
fprintf('Mejor método según R²: %s (R² = %.3f)\n', metodos{mejor_r2}, ...
        resultados_metricas.(metodos{mejor_r2}).R2);

%% ====== Generación de los Mapas de Interpolación =====%%

% Elegir el mejor método según RMSE
metodos = {'IDW','KrigingSimple','KrigingUniversal'};
[~, idx_best] = min([resultados_metricas.IDW.RMSE, ...
                     resultados_metricas.KrigingSimple.RMSE, ...
                     resultados_metricas.KrigingUniversal.RMSE]);
switch metodos{idx_best}
    case 'IDW',            best_grid = idw_grid; best_title = 'IDW (mejor RMSE)';
    case 'KrigingSimple',  best_grid = sk_grid;  best_title = 'Kriging Simple (mejor RMSE)';
    case 'KrigingUniversal', best_grid = uk_grid; best_title = 'Kriging Universal (mejor RMSE)';
end

% Verificar si tenemos conexión a internet para mapas base
try
    % Intentar acceder a un servicio de mapas
    webread('https://services.arcgisonline.com/arcgis/rest/services');
    fprintf('Conexión a servicios de mapas disponible\n');
    usar_mapa_base = true;
catch
    fprintf('Sin conexión a servicios de mapas - usando visualización básica\n');
    usar_mapa_base = false;
end

figure('Name','Mapa ICA — mejor método','Position',[200 200 900 700]);

if usar_mapa_base
    gx = geoaxes; geobasemap(gx,'streets'); hold(gx,'on');
    [lon_mesh, lat_mesh] = meshgrid(lon_grid(1,:), lat_grid(:,1));
    mask = ~isnan(best_grid);
    lat_valid = lat_mesh(mask); lon_valid = lon_mesh(mask); data_valid = best_grid(mask);
    geoscatter(gx, lat_valid, lon_valid, 12, data_valid, 'filled', ...
               'MarkerEdgeColor','none','MarkerFaceAlpha', 1);
    geoplot(gx, coords_estaciones(:,1), coords_estaciones(:,2), 'k^', ...
            'MarkerFaceColor','w','MarkerEdgeColor','k','MarkerSize',10,'LineWidth',1.2);
    geolimits(gx, [min(lat_grid(:)) max(lat_grid(:))], [min(lon_grid(:)) max(lon_grid(:))]);
    title(gx, sprintf(best_title, '%s\n%s'));
    cb = colorbar; cb.Label.String = 'ICA predicho';
else
    [~,h] = contourf(lon_grid, lat_grid, best_grid, 40, 'LineStyle','none');
    hold on;
    plot(coords_estaciones(:,2), coords_estaciones(:,1), 'k^', ...
         'MarkerFaceColor','w','MarkerEdgeColor','k','MarkerSize',10,'LineWidth',1.2);
    title(sprintf(best_title, '%s\n%s')); grid on; xlabel('Longitud'); ylabel('Latitud');
    colorbar; 
end

colormap(colores_ica);

% Título general
sgtitle('Distribución del ICA en la ZMQ según Diferentes Métodos de Interpolación', 'FontSize', 16, 'FontWeight', 'bold');

if usar_mapa_base
    fprintf('Mapas generados con mapa base y clasificación\n');
else
    fprintf('Mapas generados con ejes cartesianos y clasificación SEMARNAT\n');
end

% ================== Verficar que el modelo cargado Existe ==================
if exist('modelo_cargado_path','var') && ~isempty(modelo_cargado_path) && isfile(modelo_cargado_path)
    fprintf('Mostrando contenido del archivo cargado: %s\n', modelo_cargado_path);
    whos('-file', modelo_cargado_path);

    S = load(modelo_cargado_path, 'metricas_kfold');
    if isfield(S, 'metricas_kfold')
        disp(S.metricas_kfold);
    else
        fprintf('El archivo cargado no contiene metricas_kfold.\n');
    end
else
    fprintf('No hay modelo cargado desde archivo. Se omite inspección con whos.\n');
end

%% ======================== Validación externa (2025) ======================== %%

% Objetivo:
%   - Cargar archivos de validación 2025 (CAP, EPG, FEO) que carecen de Lat/Lon.
%   - Completar Lat/Lon por estación.
%   - Restringir al período 15-Abr-2025 a 30-Ago-2025.
%   - Evaluar ANFIS jerárquico: (SubA -> SubB -> Main) o directo Main, según diseño.
%   - Reportar métricas por estación y globales, y guardar predicciones.

fprintf('\n9. Validación con partición temporal (15-abr→15-may | 16-may→30-ago 2025)...\n');

% Archivos de validación esperados (nombres exactos):
val_archivos = {
    'ICA_datos_procesados_Validacion_CAP.xlsx', 'CAP';
    'ICA_datos_procesados_Validacion_EPG.xlsx', 'EPG';
    'ICA_datos_procesados_Validacion_FEO.xlsx', 'FEO'
};

% Coordenadas fijas por estación (como en el entrenamiento)
coords_estaciones = [
    20.5986, -100.3800; % CAP
    20.6333, -100.4000; % EPG
    20.6167, -100.4433; % FEO
];
estaciones_nombres = {'CAP','EPG','FEO'};

% Fechas de las dos ventanas
f_obs_ini = datetime(2025,4,15,0,0,0);
f_obs_fin = datetime(2025,5,15,23,59,59);
f_pred_ini = datetime(2025,5,16,0,0,0);
f_pred_fin = datetime(2025,8,30,23,59,59);

% Acumuladores
pred_obs_all = table();  % Resultados ventana Observación
pred_pred_all = table(); % Resultados ventana Predicción
metricas_part_est = table(); % Métricas por estación para ventana Predicción

% Utilitarias locales (se duplican aquí para independencia; si ya existen en sección 8, puede usarlas directamente)
function T = leer_tabla_xlsx(filepath)
    opts = detectImportOptions(filepath);
    try
        idx_dt = find(strcmpi(opts.VariableTypes,'datetime'),1);
        if ~isempty(idx_dt)
            opts = setvartype(opts, opts.VariableNames{idx_dt}, 'datetime');
        end
    catch
    end
    T = readtable(filepath, opts);
end

function T = completar_lat_lon(T, est_name, coords_estaciones, estaciones_nombres)
    if ~ismember('StationName', T.Properties.VariableNames)
        T.StationName = repmat({est_name}, height(T), 1);
    end
    if ~ismember('StationID', T.Properties.VariableNames)
        sid = find(strcmp(estaciones_nombres, est_name));
        if isempty(sid), error('StationName "%s" no reconocido.', est_name); end
        T.StationID = repmat(sid, height(T), 1);
    end
    if ~ismember('Lat', T.Properties.VariableNames)
        T.Lat = zeros(height(T),1);
    end
    if ~ismember('Lon', T.Properties.VariableNames)
        T.Lon = zeros(height(T),1);
    end
    sid = unique(T.StationID);
    if numel(sid) ~= 1
        error('Se esperaba una sola StationID en archivo de validación.');
    end
    T.Lat(:) = coords_estaciones(sid,1);
    T.Lon(:) = coords_estaciones(sid,2);
end

function [X1, X2, Yopt, fecha_vec] = preparar_entradas_valid(T)
    fecha_vars = T.Properties.VariableNames(contains(lower(T.Properties.VariableNames), 'fecha') | contains(lower(T.Properties.VariableNames), 'date') | contains(lower(T.Properties.VariableNames), 'time'));
    if isempty(fecha_vars)
        fecha_vec = NaT(height(T),1);
    else
        fecha_vec = T.(fecha_vars{1});
        if ~isdatetime(fecha_vec)
            try
                fecha_vec = datetime(fecha_vec);
            catch
                fecha_vec = NaT(height(T),1);
            end
        end
    end
    reqA = {'ICA_PM','ICA_CO','ICA_SO2','ICA_NO2','ICA_O3'};
    for i=1:numel(reqA)
        if ~ismember(reqA{i}, T.Properties.VariableNames)
            error('Falta variable requerida en validación: %s', reqA{i});
        end
    end
    X1 = T{:, reqA};
    reqB = {'TM','HR','VV','DV','Lat','Lon'};
    for i=1:numel(reqB)
        if ~ismember(reqB{i}, T.Properties.VariableNames)
            error('Falta variable requerida en validación: %s', reqB{i});
        end
    end
    X2 = T{:, reqB};
    if ismember('ICA_Total', T.Properties.VariableNames)
        Yopt = T.ICA_Total;
    else
        Yopt = [];
    end
end

function mask = en_rango(fechas, ini, fin)
    if all(isnat(fechas))
        mask = true(size(fechas));
    else
        mask = (fechas >= ini) & (fechas <= fin);
    end
end

% Iterar por archivos de validación
for iFile = 1:size(val_archivos,1)

    % Escalado geográfico para SubB en validación:
    lat_min_val = min(coords_estaciones(:,1));
    lat_max_val = max(coords_estaciones(:,1));
    lon_min_val = min(coords_estaciones(:,2));
    lon_max_val = max(coords_estaciones(:,2));

    archivo = val_archivos{iFile,1};
    est_name = val_archivos{iFile,2};
    if ~isfile(archivo)
        fprintf('No se encontró %s, se omite partición temporal.\n', archivo);
        continue;
    end

    fprintf('Procesando partición temporal para %s (%s)...\n', archivo, est_name);
    Tval = leer_tabla_xlsx(archivo);
    Tval = completar_lat_lon(Tval, est_name, coords_estaciones, estaciones_nombres);
    [X1_raw, X2_raw, Yopt_raw, fechas] = preparar_entradas_valid(Tval);

    % Máscara ventana Observación y Predicción
    mask_obs = en_rango(fechas, f_obs_ini, f_obs_fin);
    mask_pred = en_rango(fechas, f_pred_ini, f_pred_fin);

    % Extraer datos por ventana
    X1_obs = X1_raw(mask_obs,:); X2_obs = X2_raw(mask_obs,:); fechas_obs = fechas(mask_obs);
    X1_pred = X1_raw(mask_pred,:); X2_pred = X2_raw(mask_pred,:); fechas_pred = fechas(mask_pred);
    if ~isempty(Yopt_raw)
        Y_obs = Yopt_raw(mask_obs);
        Y_pred_real = Yopt_raw(mask_pred);
    else
        Y_obs = []; Y_pred_real = [];
    end

    % Limpieza de NaN
    valmask_obs = all(isfinite(X1_obs),2) & all(isfinite(X2_obs),2);
    if ~isempty(Y_obs), valmask_obs = valmask_obs & isfinite(Y_obs); end
    X1_obs = X1_obs(valmask_obs,:); X2_obs = X2_obs(valmask_obs,:); fechas_obs = fechas_obs(valmask_obs);
    if ~isempty(Y_obs), Y_obs = Y_obs(valmask_obs,:); end

    valmask_pred = all(isfinite(X1_pred),2) & all(isfinite(X2_pred),2);
    if ~isempty(Y_pred_real), valmask_pred = valmask_pred & isfinite(Y_pred_real); end
    X1_pred = X1_pred(valmask_pred,:); X2_pred = X2_pred(valmask_pred,:); fechas_pred = fechas_pred(valmask_pred);
    if ~isempty(Y_pred_real), Y_pred_real = Y_pred_real(valmask_pred,:); end

    % Si no hay datos válidos en alguna ventana, continuar
    if isempty(X1_obs)
        fprintf('Sin datos válidos en ventana Observación para %s.\n', est_name);
    end
    if isempty(X1_pred)
        fprintf('Sin datos válidos en ventana Predicción para %s.\n', est_name);
        continue;
    end

    % Evaluación jerárquica en Observación (ventana 15-abr–15-may)
    if ~isempty(X1_obs)
        % Subsistema A: contaminantes

        % Normalizar con los mismos mu/sig del entrenamiento
        X1_obs_n   = (X1_obs - escala_modelo.mu1) ./ escala_modelo.sig1;
        Y_A_obs_n  = evalfis(fisA_trained, X1_obs_n);                 % salida normalizada
        SubA_obs_n = (Y_A_obs_n - escala_modelo.muA) ./ escala_modelo.sigA;
        
        % Subsistema B: meteorológicas

        % Construir entradas de SubB (con geo_scaler, no con min/max locales)
        X2_obs_B = build_subB_inputs(X2_obs, ...
                        geo_scaler.lat_min, geo_scaler.lat_max, ...
                        geo_scaler.lon_min, geo_scaler.lon_max);
        
        % Normalizar solo las 5 primeras columnas (TM, HR, VV, DV_sin, DV_cos)
        X2_obs_B(:,1:5) = (X2_obs_B(:,1:5) - escala_modelo.mu2) ./ escala_modelo.sig2;
        Y_B_obs_n       = evalfis(fisB_trained, X2_obs_B);            % salida normalizada
        SubB_obs_n      = (Y_B_obs_n - escala_modelo.muB) ./ escala_modelo.sigB;
        
        % ANFIS principal
        X_main_obs_n = [SubA_obs_n(:), SubB_obs_n(:)];
        Y_main_obs_n = evalfis(fisMain_trained, X_main_obs_n);        % salida normalizada
        Y_main_obs   = Y_main_obs_n * escala_modelo.sigY + escala_modelo.muY; % ICA real
            
        % Construir tabla de resultados
        nobs = numel(Y_main_obs);
        pred_obs = table( ...
            repmat({est_name}, nobs,1), ...
            fechas_obs, ...
            Y_A_obs_n, ...      % <--- antes era Y_subA_obs
            Y_B_obs_n, ...      % <--- antes era Y_subB_obs
            Y_main_obs, ...
            'VariableNames', {'Estacion','FechaHora', ...
                              'Sal_Contaminantes','Sal_Meteorologicas','ICA_Predicho'});

        if ~isempty(Y_obs), pred_obs.ICA_Real = Y_obs; end
        pred_obs_all = [pred_obs_all; pred_obs]; %#ok<AGROW>
    end

    % Evaluación jerárquica en Predicción (ventana 16-may–30-ago)

    % Subsistema A
    X1_pred_n   = (X1_pred - escala_modelo.mu1) ./ escala_modelo.sig1;
    Y_A_pred_n  = evalfis(fisA_trained, X1_pred_n);
    SubA_pred_n = (Y_A_pred_n - escala_modelo.muA) ./ escala_modelo.sigA;

    % Subsistema B
    X2_pred_B = build_subB_inputs(X2_pred, ...
                    geo_scaler.lat_min, geo_scaler.lat_max, ...
                    geo_scaler.lon_min, geo_scaler.lon_max);
    X2_pred_B(:,1:5) = (X2_pred_B(:,1:5) - escala_modelo.mu2) ./ escala_modelo.sig2;
    Y_B_pred_n       = evalfis(fisB_trained, X2_pred_B);
    SubB_pred_n      = (Y_B_pred_n - escala_modelo.muB) ./ escala_modelo.sigB;
    
    % ANFIS Principal
    X_main_pred_n = [SubA_pred_n(:), SubB_pred_n(:)];
    Y_main_pred_n = evalfis(fisMain_trained, X_main_pred_n);
    Y_main_pred   = Y_main_pred_n * escala_modelo.sigY + escala_modelo.muY;

    % Creación de la tabla
    np = numel(Y_main_pred);
    pred_pred = table( ...
        repmat({est_name}, np,1), ...
        fechas_pred, ...
        Y_A_pred_n, ...
        Y_B_pred_n, ...
        Y_main_pred, ...
        'VariableNames', {'Estacion','FechaHora', 'Sal_Contaminantes','Sal_Meteorologicas','ICA_Predicho'});

    if ~isempty(Y_pred_real), pred_pred.ICA_Real = Y_pred_real; end
    pred_obs_all = table();  % Importante: table()
    pred_pred_all = table(); % Importante: table()
    metricas_part_est = table();

    % Métricas de la ventana Predicción por estación
    RMSE = NaN; MAE = NaN; R2 = NaN;
    if ismember('ICA_Real', pred_pred.Properties.VariableNames)
        Ytrue = pred_pred.ICA_Real; Yhat = pred_pred.ICA_Predicho;
        if ~isempty(Ytrue)
            RMSE = sqrt(mean((Ytrue - Yhat).^2));
            MAE  = mean(abs(Ytrue - Yhat));
            if std(Ytrue) > 0 && std(Yhat) > 0
                R2 = corr(Ytrue, Yhat)^2;
            end
        end
    end
    metricas_part_est = [metricas_part_est; table({est_name}, RMSE, MAE, R2, ...
        'VariableNames', {'Estacion','RMSE','MAE','R2'})]; %#ok<AGROW>

    % Exportar archivos por estación
    out_dir = 'resultados_validacion';
    if ~exist(out_dir,'dir'), mkdir(out_dir); end
    ts_stamp = datestr(now,'yyyymmdd_HHMMSS');
    if exist('ts_stamp','var')==0 || isempty(ts_stamp)
        ts_stamp = datestr(now,'yyyymmdd_HHMMSS'); % fallback
    end
    writetable(pred_pred, fullfile(out_dir, sprintf('Predicciones_Particion_%s_%s.xlsx', est_name, ts_stamp)));
    if ~isempty(X1_obs)
        writetable(pred_obs, fullfile(out_dir, sprintf('Observacion_Particion_%s_%s.xlsx', est_name, ts_stamp)));
    end
    fprintf('Archivos partición temporal generados para %s.\n', est_name);
end

% Métricas globales de la ventana Predicción
if ~isempty(pred_pred_all) && ismember('ICA_Real', pred_pred_all.Properties.VariableNames)
    Yg = pred_pred_all.ICA_Real; Yh = pred_pred_all.ICA_Predicho;
    if ~isempty(Yg)
        RMSEg = sqrt(mean((Yg - Yh).^2));
        MAEg  = mean(abs(Yg - Yh));
        if std(Yg)>0 && std(Yh)>0
            R2g = corr(Yg, Yh)^2;
        else
            R2g = NaN;
        end
    else
        RMSEg = NaN; MAEg = NaN; R2g = NaN;
    end
else
    RMSEg = NaN; MAEg = NaN; R2g = NaN;
end
metricas_part_global = table(RMSEg, MAEg, R2g);

%% ====== Resultados Promedio de Métricas ====== %%

rmse_vec = []; mae_vec = []; r2_vec = [];
K_print = [];

% Intentar desde variables de la sesión
if exist('K','var'), K_print = K; end
if exist('rmse_total','var') && ~isempty(rmse_total), rmse_vec = rmse_total(:); end
if exist('mae_total','var')  && ~isempty(mae_total),  mae_vec  = mae_total(:);  end
if exist('r2_total','var')   && ~isempty(r2_total),   r2_vec   = r2_total(:);   end

% Si están vacíos, intentar desde el modelo cargado (metricas_kfold)
if (isempty(rmse_vec) || isempty(mae_vec) || isempty(r2_vec) || isempty(K_print)) ...
        && exist('modelo_cargado_path','var') && ~isempty(modelo_cargado_path) && isfile(modelo_cargado_path)
    try
        Srep = load(modelo_cargado_path, 'metricas_kfold');
        if isfield(Srep,'metricas_kfold') && ~isempty(Srep.metricas_kfold)
            mk = Srep.metricas_kfold;
            if isempty(K_print) && isfield(mk,'K') && ~isempty(mk.K), K_print = mk.K; end
            if isempty(rmse_vec) && isfield(mk,'rmse_por_fold') && ~isempty(mk.rmse_por_fold)
                rmse_vec = mk.rmse_por_fold(:);
            end
            if isempty(mae_vec) && isfield(mk,'mae_por_fold') && ~isempty(mk.mae_por_fold)
                mae_vec = mk.mae_por_fold(:);
            end
            if isempty(r2_vec) && isfield(mk,'r2_por_fold') && ~isempty(mk.r2_por_fold)
                r2_vec = mk.r2_por_fold(:);
            end
        end
    catch
    end
end

% Encabezado sin depender de K
if ~isempty(K_print)
    fprintf('\n=== RESULTADOS PROMEDIO (K=%d) ===\n', K_print);
else
    fprintf('\n=== RESULTADOS PROMEDIO (K no disponible) ===\n');
end

% Impresión de métricas con defensas
if ~isempty(rmse_vec)
    fprintf('RMSE: prom = %.4f | std = %.4f | n = %d\n', mean(rmse_vec), std(rmse_vec), numel(rmse_vec));
else
    fprintf('RMSE: no disponible\n');
end
if ~isempty(mae_vec)
    fprintf('MAE:  prom = %.4f | std = %.4f | n = %d\n', mean(mae_vec), std(mae_vec), numel(mae_vec));
else
    fprintf('MAE:  no disponible\n');
end
if ~isempty(r2_vec)
    fprintf('R^2:  prom = %.4f | std = %.4f | n = %d\n', mean(r2_vec), std(r2_vec), numel(r2_vec));
else
    fprintf('R^2:  no disponible\n');
end

% Resultados Globales
if exist('K','var')
    fprintf('\n=== RESULTADOS PROMEDIO (K=%d) ===\n', K);
elseif exist('modelo_cargado_path','var') && isfile(modelo_cargado_path)
    try
        Sx = load(modelo_cargado_path,'metricas_kfold');
        if isfield(Sx,'metricas_kfold') && isfield(Sx.metricas_kfold,'K') && ~isempty(Sx.metricas_kfold.K)
            fprintf('\n=== RESULTADOS PROMEDIO (K=%d) ===\n', Sx.metricas_kfold.K);
        else
            fprintf('\n=== RESULTADOS PROMEDIO (K no disponible) ===\n');
        end
    catch
        fprintf('\n=== RESULTADOS PROMEDIO (K no disponible) ===\n');
    end
else
    fprintf('\n=== RESULTADOS PROMEDIO (K no disponible) ===\n');
end

%% ====== Generando visualizaciones de errores finales ====== %%

% Recuperación de Datos
datos_finales = table();
origen_datos = '';

% Buscar en Memoria
if exist('pred_pred_all', 'var') && ~isempty(pred_pred_all)
    datos_finales = pred_pred_all;
    origen_datos = 'Memoria (Validación Temporal)';
elseif exist('resultados_all', 'var') && ~isempty(resultados_all)
    datos_finales = resultados_all;
    origen_datos = 'Memoria (Validación Cruzada)';
end

% Si falla memoria, buscar en Disco
if isempty(datos_finales)
    out_dir = 'resultados_validacion';
    if exist(out_dir, 'dir')
        files = dir(fullfile(out_dir, 'Predicciones_Particion_*.xlsx'));
        if ~isempty(files)
            fprintf('⚠ Datos no en memoria. Recuperando %d archivos de disco...\n', length(files));
            for i = 1:length(files)
                try
                    t = readtable(fullfile(files(i).folder, files(i).name));
                    % Normalizar nombres si es necesario
                    if ismember('StationName', t.Properties.VariableNames) && ~ismember('Estacion', t.Properties.VariableNames)
                        t.Estacion = t.StationName; 
                    end
                    datos_finales = [datos_finales; t]; %#ok<AGROW>
                catch
                end
            end
            origen_datos = 'Recuperado de Disco';
        end
    end
end

% Generación de gráficas
if ~isempty(datos_finales) && ismember('ICA_Real', datos_finales.Properties.VariableNames)
    
    % Preparar vectores limpios
    Y_real = datos_finales.ICA_Real;
    Y_pred = datos_finales.ICA_Predicho;
    if ismember('Estacion', datos_finales.Properties.VariableNames)
        Estacion = datos_finales.Estacion;
    else
        Estacion = repmat({'Desconocida'}, height(datos_finales), 1);
    end
    
    % Filtrar NaNs
    mask = isfinite(Y_real) & isfinite(Y_pred);
    Y_real = Y_real(mask);
    Y_pred = Y_pred(mask);
    Estacion = Estacion(mask);
    
    if ~isempty(Y_real)
        % Calcular Residuos
        residuals = Y_pred - Y_real;
        
        fprintf('Datos cargados desde: %s. Generando 3 gráficas...\n', origen_datos);

        % --- GRÁFICA 1: ANÁLISIS DE RESIDUOS (Histograma Gaussiano) ---
        figure('Name', 'Robustez: Residuos', 'Color', 'w', 'Position', [100 100 600 400]);
        histogram(residuals, 50, 'Normalization', 'pdf', 'FaceColor', '#2980b9', 'EdgeColor', 'none');
        hold on;
        % Curva normal teórica
        x_range = linspace(min(residuals), max(residuals), 100);
        mu = mean(residuals); sig = std(residuals);
        plot(x_range, normpdf(x_range, mu, sig), 'r-', 'LineWidth', 2);
        
        xlabel('Error (Predicho - Real)', 'FontWeight', 'bold');
        title({'\bf Distribución del Error (Validación de Aleatoriedad)', ...
               sprintf('Media=%.2f | Std=%.2f', mu, sig)});
        legend('Frecuencia de Error', 'Distribución Normal Ideal');
        grid on;

        % --- GRÁFICA 2: BLAND-ALTMAN ---
        figure('Name', 'Robustez: Bland-Altman', 'Color', 'w', 'Position', [700 100 600 400]);
        promedios = (Y_pred + Y_real) / 2;
        
        scatter(promedios, residuals, 20, 'filled', 'MarkerFaceColor', '#8e44ad', 'MarkerFaceAlpha', 0.4);
        hold on;
        yline(mu, 'k-', 'LineWidth', 2, 'Label', 'Sesgo Medio');
        yline(mu + 1.96*sig, 'r--', 'LineWidth', 2, 'Label', '+1.96 SD');
        yline(mu - 1.96*sig, 'r--', 'LineWidth', 2, 'Label', '-1.96 SD');
        
        xlabel('Valor Promedio (ICA)', 'FontWeight', 'bold');
        ylabel('Diferencia (Predicho - Real)', 'FontWeight', 'bold');
        title('\bf Gráfico de Bland-Altman: Consistencia del Error');
        grid on;

        % --- GRÁFICA 3: BOXPLOT POR ESTACIÓN ---
        figure('Name', 'Robustez: Por Estación', 'Color', 'w', 'Position', [400 550 600 400]);
        boxplot(abs(residuals), Estacion);
        xlabel('Estación de Monitoreo', 'FontWeight', 'bold');
        ylabel('Error Absoluto (MAE)', 'FontWeight', 'bold');
        title('\bf Distribución del Error por Zona Geográfica');
        grid on;
        
        fprintf('¡Éxito! Gráficas generadas correctamente.\n');
    else
        fprintf('Los datos cargados contienen solo NaNs. Revise la validación.\n');
    end
else
    fprintf('ERROR CRÍTICO: No se encontraron datos ni en memoria ni en la carpeta "resultados_validacion".\n');
    fprintf('Ejecute primero la Sección 9 para generar predicciones.\n');
end


% ========================================================================
% FUNCIONES AUXILIARES
% ========================================================================

% Carga datos de múltiples archivos Excel con ajuste de datetime a hora exacta.    
function stations_data = cargar_datos_estacion(data_files)
    
    stations_data = {};
    
    for i = 1:size(data_files, 1)
        try

            % Lee la tabla del archivo de Excel.
            opts = detectImportOptions(data_files{i,1});
            opts = setvartype(opts, 'datetime', 'datetime');
            tbl = readtable(data_files{i,1}, opts);
            
            % Buscar la columna datetime en la tabla
            datetime_col_idx = [];
            for col = 1:width(tbl)
                if isdatetime(tbl{:, col})
                    datetime_col_idx = col;
                    break; % Solo procesamos la primera columna datetime encontrada
                end
            end
            
            if ~isempty(datetime_col_idx)
                fprintf('Procesando ajuste de datetime en columna %d de %s...\n', ...
                        datetime_col_idx, data_files{i,1});
                
                % Obtener el vector de datetime
                datetime_vector = tbl{:, datetime_col_idx};
                
                % Aplicar el ajuste de redondeo a la hora exacta más cercana
                datetime_ajustado = ajustar_datetime_hora_exacta(datetime_vector);
                
                % Reemplazar la columna original con los valores ajustados
                tbl{:, datetime_col_idx} = datetime_ajustado;
                
                fprintf('Ajuste completado: %d registros procesados.\n', length(datetime_vector));
            else
                fprintf('No se encontraron columnas datetime en %s\n', data_files{i,1});
            end
            
            % Se verifica si la columna 'Categoria' existe en la tabla recién cargada.
            % 'ismember' comprueba si el texto 'Categoria' está en la lista de nombres de variables.
            if ismember('Categoria', tbl.Properties.VariableNames)
                % Si la columna existe, se elimina asignándole un valor vacío [].
                tbl.Categoria = [];
                fprintf('Columna ''Categoria'' eliminada de %s para estandarizar.\n', data_files{i,1});
            end
            
            % Se añaden los identificadores de la estación.
            tbl.StationName = repmat({data_files{i,2}}, height(tbl), 1);
            tbl.StationID = repmat(i, height(tbl), 1);
            
            stations_data{end+1} = tbl;
            fprintf(' -> Archivo %s cargado correctamente.\n', data_files{i,1});
            
        catch e
            fprintf(' -> Error al cargar %s: %s\n', data_files{i,1}, e.message);
        end
    end

end

% Ajuste de DATETIME
function datetime_ajustado = ajustar_datetime_hora_exacta(datetime_vector)

    % Ajusta registros de datetime a la hora exacta más cercana
    % Ejemplo: 25/03/2022 3:59:59.999 se convierte en 25/03/2022 4:00:00
    %
    % ENTRADA:
    %   datetime_vector: Vector de objetos datetime
    %
    % SALIDA:
    %   datetime_ajustado: Vector de datetime con horas ajustadas
    
    % Inicializar el vector de salida
    datetime_ajustado = datetime_vector;
    
    % Contadores para seguimiento de cambios
    registros_ajustados = 0;
    
    % Procesar cada elemento del vector datetime
    for i = 1:length(datetime_vector)
        if ~isnat(datetime_vector(i))  % Verificar que no sea NaT (Not a Time)
            
            % Extraer componentes del datetime original
            fecha_original = datetime_vector(i);
            
            % Obtener los segundos y milisegundos
            segundos = second(fecha_original);
            
            % Criterio de ajuste: si los segundos son >= 59.5, redondear a la hora siguiente
            % Esto captura casos como 3:59:59.999, 3:59:59.500, etc.
            if segundos >= 59.5
                % Crear nuevo datetime redondeando a la hora siguiente
                % Método 1: Añadir 1 minuto y luego redondear hacia abajo
                fecha_temp = fecha_original + minutes(1);
                
                % Redondear hacia abajo eliminando segundos y milisegundos
                datetime_ajustado(i) = dateshift(fecha_temp, 'start', 'minute');
                
                registros_ajustados = registros_ajustados + 1;
                
            elseif segundos >= 0.5 && segundos < 59.5
                % Para segundos entre 0.5 y 59.4, redondear al minuto más cercano
                if segundos >= 30
                    % Redondear al siguiente minuto
                    fecha_temp = fecha_original + minutes(1);
                    datetime_ajustado(i) = dateshift(fecha_temp, 'start', 'minute');
                    registros_ajustados = registros_ajustados + 1;
                else
                    % Redondear hacia abajo al minuto actual
                    datetime_ajustado(i) = dateshift(fecha_original, 'start', 'minute');
                    if segundos >= 0.5  % Solo contar como ajuste si había decimales
                        registros_ajustados = registros_ajustados + 1;
                    end
                end
            else
                % Para segundos < 0.5, mantener el minuto actual pero eliminar decimales
                datetime_ajustado(i) = dateshift(fecha_original, 'start', 'minute');
                if segundos > 0  % Contar como ajuste si había segundos
                    registros_ajustados = registros_ajustados + 1;
                end
            end
        end
    end
    
    % Reportar estadísticas del ajuste
    if registros_ajustados > 0
        fprintf('%d registros fueron ajustados a la hora/minuto exacto.\n', registros_ajustados);
        
        % Mostrar algunos ejemplos de ajustes (máximo 3)
        ejemplos_mostrados = 0;
        for i = 1:length(datetime_vector)
            if ejemplos_mostrados >= 3
                break;
            end
            
            if datetime_vector(i) ~= datetime_ajustado(i)
                fprintf('   -> Ejemplo: %s -> %s\n', ...
                        char(datetime_vector(i)), char(datetime_ajustado(i)));
                ejemplos_mostrados = ejemplos_mostrados + 1;
            end
        end
    else
        fprintf('No se requirieron ajustes de datetime.\n');
    end

end

function unified_data = unificar_datos(stations_data)

    if isempty(stations_data)
        unified_data = table;
        return;
    end

    % Encontrar las columnas comunes a TODAS las tablas.
    % Se empieza con los nombres de las columnas de la primera tabla.
    common_vars = stations_data{1}.Properties.VariableNames;
    
    % Se itera sobre las demás tablas para encontrar la intersección de columnas.
    for i = 2:length(stations_data)
        % 'intersect' devuelve los nombres que están en ambas listas.
        common_vars = intersect(common_vars, stations_data{i}.Properties.VariableNames, 'stable');
    end
    fprintf('Se encontraron %d columnas comunes en todos los archivos.\n', length(common_vars));

    % Crear una nueva lista de tablas, pero seleccionando solo las columnas comunes.
    tables_to_stack = cell(size(stations_data));
    for i = 1:length(stations_data)
        % Se crea una nueva tabla temporal solo con las columnas en 'common_vars'.
        tables_to_stack{i} = stations_data{i}(:, common_vars);
    end

    % Unir (concatenar) las tablas ya estandarizadas.
    % Esta línea ahora funcionará sin error porque todas las tablas tienen las mismas columnas.
    unified_data = vertcat(tables_to_stack{:});

    % El resto de la función original se mantiene igual.
    % Eliminar filas con valores NaN para asegurar la calidad de los datos.
    unified_data = rmmissing(unified_data);

    % Añadir coordenadas (se usan las reales en la sección de interpolación).
    latitudes = [20.5986, 20.6333, 20.6167]; % CAP, EPG, FEO
    longitudes = [-100.3800, -100.4000, -100.4433]; % CAP, EPG, FEO
    
    % Se crea un mapeo de StationID a coordenadas.
    station_ids = unique(unified_data.StationID);
    lat_map = containers.Map(station_ids, latitudes(station_ids));
    lon_map = containers.Map(station_ids, longitudes(station_ids));
    
    % Se asignan las coordenadas a cada fila según su StationID.
    % La función 'values' devuelve un cell array. Usamos 'cell2mat' para convertirlo
    % a un vector numérico estándar, que es lo que la tabla necesita.
    % La transposición (') final asegura que sea un vector columna.
    lat_vector = cell2mat(values(lat_map, num2cell(unified_data.StationID)));
    lon_vector = cell2mat(values(lon_map, num2cell(unified_data.StationID)));

    % Añadir los vectores numéricos como nuevas columnas a la tabla.
    % Ahora las dimensiones y el tipo de dato son correctos.
    unified_data.Lat = lat_vector;
    unified_data.Lon = lon_vector;
end

% Prepara variables de entrada y salida para ANFIS
function [X_contaminantes, X_meteorologicas, Y_target] = prepare_anfis_variables(unified_data)
    X_contaminantes   = unified_data{:, {'ICA_PM', 'ICA_CO', 'ICA_SO2', 'ICA_NO2', 'ICA_O3'}};
    X_meteorologicas  = unified_data{:, {'TM', 'HR', 'VV', 'DV', 'Lat', 'Lon'}};
    Y_target          = unified_data.ICA_Total;
    fprintf('Datos cargados: %d muestras con %d variables de entrada.\n', size(unified_data,1), size(X_contaminantes,2) + size(X_meteorologicas,2));
end

% División 70% entrenamiento, 30% validación
function [X1_train, X1_val, X2_train, X2_val, Y_train, Y_val] = partition_data(X_contaminantes, X_meteorologicas, Y_target, unified_data)
    
    % División temporal 70% entrenamiento (pasado), 30% validación (futuro)
    unified_data = sortrows(unified_data, 'datetime');
    n = height(unified_data);
    cut = floor(0.7*n);
    
    idx_train = false(n,1);
    idx_val   = false(n,1);
    
    idx_train(1:cut) = true;
    idx_val(cut+1:end) = true;
    
    % Se usan los índices para dividir las variables de entrada y salida.
    % Las variables ya están en el formato correcto de matriz/vector.
    X1_train = X_contaminantes(idx_train, :);
    X1_val = X_contaminantes(idx_val, :);

    X2_train = X_meteorologicas(idx_train, :);
    X2_val = X_meteorologicas(idx_val, :);

    Y_train = Y_target(idx_train);
    Y_val = Y_target(idx_val);

    fprintf(' Datos divididos: %d para entrenamiento, %d para validación.\n', sum(idx_train), sum(idx_val));
end

function X2_B = build_subB_inputs(X2_raw, lat_min, lat_max, lon_min, lon_max)
    % X2_raw: [TM, HR, VV, DV_deg, Lat, Lon] (6 columnas)
    % Devuelve: [TM, HR, VV, DV_sin, DV_cos, Lat_g, Lon_g] (7 columnas)

    if isempty(X2_raw)
        X2_B = X2_raw;
        return;
    end

    % DV en grados -> radianes
    DV_deg = X2_raw(:,4);
    DV_rad = deg2rad(DV_deg);
    DV_sin = sin(DV_rad);
    DV_cos = cos(DV_rad);

    Lat = X2_raw(:,5);
    Lon = X2_raw(:,6);

    Lat_g = (Lat - lat_min) ./ (lat_max - lat_min);
    Lon_g = (Lon - lon_min) ./ (lon_max - lon_min);

    Lat_g = max(0, min(1, Lat_g));
    Lon_g = max(0, min(1, Lon_g));

    X2_B = [X2_raw(:,1:3), DV_sin, DV_cos, Lat_g, Lon_g];
end