

% Define parameters
output_directory = 'D:\DHPSFU\2025-03-26_Fusion_PSF'; % Directory to save the output file
output_filename = 'Calib.tif'; % Simulated Calibration TIFF file
output_filename2 = 'Bead_30nm_shift.tif'; % Sub-pixel shifted beads for simulation.

% Create output directory if it does not exist
if ~exist(output_directory, 'dir')
    mkdir(output_directory);
end

% Create random sub-pixel shift (3 decimal place)
x = round((rand(5000, 1) * 2 -1) * 1000) / 1000;
y = round((rand(5000, 1) * 2 -1) * 1000) / 1000;
z = round((rand(5000, 1) * 3.9627 -1.98135) * 1000) / 1000;  % Microscope 1
%z = round((rand(5000, 1) * 4.8 -2.4) * 1000) / 1000; % Microscope 2
% cal=4.9386; % Scaling factor, Microscope 1
cal=9.17; % Scaling factor, Microscope 2

frame_numbers2 = 1:length(x);
% Initialize image stack
stack_initialized = false;
output_path2 = fullfile(output_directory, output_filename2);
for i = 1:length(x)
    im = PSF_generator(maskRec,[x(i)/cal,y(i)/cal,0.15],IS,z(i),opt_phase_mat,g_bfp,circ,circ_sc,int_cos,vec_model_flag); % Introduce random lateral shifts to the PSFs at different Z.
    % im = im + im.*randn(63).*0;
    img = uint16(20000 * mat2gray(im));
    img2 = uint16(img);
    imwrite(img2,output_path2, 'tif', 'WriteMode','append','Compression','none');
    % imagesc(im1-im2)
end

% Prepare data
data2 = [frame_numbers2', x, y, z];

% Save to CSV
output_filename3 = 'GT_shift.csv'; % Save the shifts introduced to the PSFs as the ground truth. 
csvfile2 = fullfile(output_directory, output_filename3);
headers2 = {'Frame', 'X', 'Y', 'Z'};  % Column headers

% Write to CSV file
fid2 = fopen(csvfile2, 'w');  % Open file for writing
fprintf(fid2, '%s,%s,%s,%s\n', headers2{:});  % Write headers
fclose(fid2);
% Append data
dlmwrite(csvfile2, data2, '-append');
disp(['Data saved to ', output_filename3]);


% Define the range for xyz(3)
z_range = -1.9647:0.0333:1.9647; % Microscope 1
%z_range = -2.4:0.06:2.4;  % Microscope 2

% Initialize parameters for PSF_generator
phase_mask = maskRec; % Define or load your phase mask
xyz = [0/cal, 0/cal, 0.15]; % Initial xyz value
z_range_gt = 0.06:0.06:4.86;  % Range for z values (GT)
x_value = 0;           % Constant x value
y_value = 0;           % Constant y value
frame_numbers = 1:length(z_range_gt);  % Frame numbers based on z_range

% Loop through z_range to generate the image stack
for z = z_range
    % Update the z-coordinate
    Iimg = PSF_generator(phase_mask, xyz, IS, z, opt_phase_mat, g_bfp, circ, circ_sc, int_cos, vec_model_flag);

    % Convert image to uint16 for saving
    Iimg_uint16 = uint16(20000 * mat2gray(Iimg)); % Normalize to [0, 65535]
     Iimg_uint8 = uint8(255 * mat2gray(Iimg)); % Scale image to [0, 255]

    % Convert uint8 to uint16 by scaling
    % Iimg_uint16 = uint16(Iimg_uint8);

    % Save to TIF file
    output_path = fullfile(output_directory, output_filename);
    if ~stack_initialized
        imwrite(Iimg_uint16, output_path, 'tif', 'Compression', 'none');
        stack_initialized = true;
    else
        imwrite(Iimg_uint16, output_path, 'tif', 'WriteMode', 'append', 'Compression', 'none');
    end
end
% 
 
% % Prepare data
data = [frame_numbers', repmat(x_value, length(frame_numbers), 1), ...
        repmat(y_value, length(frame_numbers), 1), z_range_gt'];

% % Save to CSV
output_filename2 = 'GT.csv'; % Name of the CSV file
csvfile = fullfile(output_directory, output_filename2);
headers = {'Frame', 'X', 'Y', 'Z'};  % Column headers
% 
% % Write to CSV file
fid = fopen(csvfile, 'w');  % Open file for writing
fprintf(fid, '%s,%s,%s,%s\n', headers{:});  % Write headers
fclose(fid);
% Append data
dlmwrite(csvfile, data, '-append');
disp(['Data saved to ', output_filename]);