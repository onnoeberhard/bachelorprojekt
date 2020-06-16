clear all; close all; clc;

% TODO: Change port accordingly, e.g. 'COM3' or '/dev/tty.usbserial-A106PX8C'
port = '/dev/tty.usbserial-A106PX8C';
arduino = serialport(port, 9600);

% Global variables
global run cali offset_m offset_b off_count X Y c;

% Control variables
run = true;
cali = false;
offset_m = 0;
offset_b = 0;
off_count = 0;

X = [];    % Matrix to store measurements
T = 40;    % Amount of datapoints to use for convolution / smoothing (T/2 seconds setup time!)
Y = [];    % Matrix to store smoothed values

% Window function for smoothing
w = linspace(0, 1, T);
w = w / sum(w);    % Normalization, the sum is theoretically T/2, but it needs to be exact

% Parameters for 100ms integration
k = 0.07300424260104722;
b = -21.70479394467475;
t0 = -36.5046091840224;    % This is the delay between our measurement and MIREX (response time in 500ms -> ca. 20s)

% Import reference data
ref = readtable('reference_100ms.csv');
ref = ref{:, 2:end};

% Setup figure with calibrate and reset button
c = uicontrol('Style', 'togglebutton', 'String', 'Calibrate', 'Callback', @calibrate, 'Position', [10 10 100 20]);
r = uicontrol('String', 'Reset Calibration', 'Callback', @reset, 'Position', [120 10 100 20]);
q = uicontrol('String', 'Stop Measurement', 'Callback', @stop, 'Position', [230 10 100 20]);
h = uicontrol('String', 'Clear History', 'Callback', @clear, 'Position', [340 10 100 20]);

while run
    % Read data from serial port
    x = strip(arduino.readline());
    disp(x);
    x = str2num(x);
    X = [X ; x];
    
    % Compute offset when in calibration mode
    if cali
        off_count = off_count + 1;
        offset_m = mean(X(end-off_count+1:end, 3)) - mean(ref(1:200, 2));
        offset_b = mean(X(end-off_count+1:end, 2)) - mean(ref(1:200, 1));
    end
    
    % Superimpose current measurements onto reference data
    subplot(3, 4, 9);
    plot(ref(:, 1), '.')
    yline(x(2), 'r');
    if offset_b ~= 0
        yline(x(2) - offset_b, 'g');
    end
    xlim([0 inf])
    title('off');
    subplot(3, 4, 10);
    histogram(ref(:, 1), 'Normalization', 'probability');
    hold on;
    histogram(X(:, 2), 'Normalization', 'probability');
    if offset_b ~= 0
        histogram(X(:, 2) - offset_b, 'Normalization', 'probability');
    end
    hold off;
    title('off hist');
    subplot(3, 4, 11);
    plot(ref(:, 2), '.');
    yline(x(3), 'r');
    if offset_m ~= 0
        yline(x(3) - offset_m, 'g');
    end
    xlim([0 inf])
    title('on');
    subplot(3, 4, 12);
    histogram(ref(:, 2), 'Normalization', 'probability');
    hold on;
    histogram(X(:, 3), 'Normalization', 'probability');
    if offset_m ~= 0
        histogram(X(:, 3) - offset_m, 'Normalization', 'probability');
    end
    hold off;
    title('on hist');
    
    % At least T measurements are needed to start evaluation
    if length(X) < T
        continue;
    end
    
    % Smooth measurements
    z1 = sum(X(end-T+1:end, 3)' .* w);                                    % Smoothed measurement
    z2 = sum(X(end-T+1:end, 2)' .* w);                                    % Smoothed background
    Y = [Y ; [x(1), z1, z2]];                                             % Store newly smoothed values
    
    % Calculate dB values. For greater efficiency, this matrix could be
    % updated row by row for every new measurement but would have to be
    % recomputed every time the offset values are changed.
    Z = [Y(:, 1), (Y(:, 2) - offset_m - (Y(:, 3) - offset_b))*k + b];
    
    % Plot dB values
    if length(Z) < 120    % Plot everything in one graph until 60s have passed
        subplot(3, 4, 1:8);
        plot(Z(:, 1)/1000, Z(:, 2));
        xlim([20 inf])
        title('[dB]')
    else                  % After 60s plot complete history and last minute separately
        subplot(3, 4, 1:4);
        plot(Z(end-120+1:end, 1)/1000, Z(end-120+1:end, 2));
        title('Last minute [dB]')
        xlim([Z(end-120+1, 1)/1000 inf])
        subplot(3, 4, 5:8);
        plot(Z(:, 1)/1000, Z(:, 2));
        xlim([20 inf])  
        title('Compete history [dB]')
    end
end

function calibrate(~, ~)
    global cali off_count c;
    cali = ~cali;
    if cali
        c.String = 'Calibrating...';
        off_count = 0;
    else
        c.String = 'Calibrate';
    end
end

function reset(~, ~)
    global offset_b offset_m;
    offset_b = 0;
    offset_m = 0;
end

function stop(~, ~)
    global run;
    run = false;
end

function clear(~, ~)
    global X Y;
    X = [];
    Y = [];
end