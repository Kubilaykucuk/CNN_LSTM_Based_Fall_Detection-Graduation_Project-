tiradar = mmWaveRadar("TI IWR6843ISK");
tiradar.AzimuthLimits = [-60 60];
tiradar.DetectionCoordinates = "Sensor spherical";
pause(1);

frameNum = 0;
zOffset = 1.0;

while true
    [objDetsRct, timestamp, ~, ~] = tiradar();
    numDets = numel(objDetsRct);

    if numDets == 0
        pause(0.05);
        continue;
    end

    frameNum = frameNum + 1;

    % Preallocate arrays
    xpos = zeros(numDets, 1);
    ypos = zeros(numDets, 1);
    zpos = zeros(numDets, 1);
    postures = strings(numDets, 1);

    for i = 1:numDets
        meas = objDetsRct{i}.Measurement;
        if numel(meas) < 3
            continue;
        end
        xpos(i) = meas(1);
        ypos(i) = meas(2);
        zpos(i) = meas(3) + zOffset;

        % Basic posture classification based on Z height
        if zpos(i) > 0.9
            postures(i) = "Standing";
        elseif zpos(i) > 0.4
            postures(i) = "Sitting";
        else
            postures(i) = "Fallen";
        end
    end

    % === Construct frame structure vectorized ===
    frameStruct = struct( ...
        'Frame', repmat(frameNum, numDets, 1), ...
        'Timestamp', repmat(timestamp, numDets, 1), ...
        'ObjectID', (1:numDets)', ...
        'X_m_', xpos, ...
        'Y_m_', ypos, ...
        'Z_m_', zpos, ...
        'Vx_m_s_', zeros(numDets, 1), ...
        'Vy_m_s_', zeros(numDets, 1), ...
        'Vz_m_s_', zeros(numDets, 1), ...
        'Vr_m_s_', zeros(numDets, 1), ...
        'range_sc_m_', sqrt(xpos.^2 + ypos.^2 + zpos.^2), ...
        'azimuth_sc_rad_', atan2(ypos, xpos), ...
        'azimuth_sc_deg_', atan2d(ypos, xpos), ...
        'Posture', postures ...
    );

    % === Encode as JSON and print for Python to read ===
    jsonStr = jsonencode(frameStruct);
    fprintf('%s\n', jsonStr);

    pause(0.05);  % allow time for Python to consume
end