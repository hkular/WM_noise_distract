function [image_final] = FilteredGratings(n, p,t, TrialStuff)
% Make a set of filtered images (load white noise filtered in the frequency domain). 
% Each image has broadband spatial frequency content, and orientation 
% content varying between 0-180 deg.

% taken from MMH April 2020
% adapted HK June 2022
    %% Set up parameters here

    % make sure we're using the expt random seed 
    rng(t.MySeed);
    
    % size of the final image in pixels
    final_size_pix = p.PatchSize;
 
         %% make white noise image to filter
         % start with a meshgrid
            X=-0.5*p.PatchSize+.5:1:.5*p.PatchSize-.5; Y=-0.5*p.PatchSize+.5:1:.5*p.PatchSize-.5;
            [x,y] = meshgrid(X,Y);

            whitenoise_sine = (sin(p.SF/p.ppd*2*pi*(y.*sin(0*pi/180)+x.*cos(0*pi/180))));
            sine_whitenoisecontrast = std(whitenoise_sine(:));
            WhiteNoiseAreHere = ones(p.PatchSize,p.PatchSize) * p.MyGrey;

            % create white noise
            %Make uniform noise, put it into fourrier space, make sf filer
            noise = rand(p.PatchSize,p.PatchSize)*2-1;
            fn_noise = fftshift(fft2(noise));
            sfFilter = Bandpass2([p.PatchSize p.PatchSize], p.Noise_fLow/p.fNyquist, p.Noise_fHigh/p.fNyquist);
            %Get rid of gibbs ringing artifacts
            smoothfilter = fspecial('gaussian', 10, 4);   % make small gaussian blob
            sfFilter = filter2(smoothfilter, sfFilter); % convolve smoothing blob w/ s.f. filter
            %Bring noise back into real space
            filterednoise = real(ifft2(ifftshift(sfFilter.*fn_noise)));
            %Scale the contrast of the noise back up (it's lost some in the fourier
            %domain) by relating it to the contrast of the grating distractor (before gaussian was applied)
            current_noise_contrast = std(filterednoise(:));
            scaling_factor = sine_whitenoisecontrast/current_noise_contrast;
            filterednoise = filterednoise*scaling_factor;
            %store it
            WhiteNoiseAreHere(:,:) = max(0,min(255,p.MyGrey+p.MyGrey*(p.whitenoiseContrast * filterednoise)));
            
            image = WhiteNoiseAreHere;
          
        %% Define spectral domain grid/sampling rate etc
        % first, define sampling rate and frequency axis
        samp_rate_pix = 1;   % samples per pixel, always one.
        %samp_rate_pix = 2;
        % get frequency axis
        nframes = final_size_pix;
        nyq = .5*samp_rate_pix;         
        % step size (or resolution) in the frequency domain - depends on sampling rate and length of the signal
        freq_step_cpp = samp_rate_pix/nframes;  
        % x-axis in freq domain, after fftshift
        fax = -nyq:freq_step_cpp:nyq-freq_step_cpp;
        center_pix = find(abs(fax)==min(abs(fax)));

        % next, we're listing the frequencies corresponding to each point in
        % the FFT representation, in grid form
        [ x_freq, y_freq ] = meshgrid(fax,fax);
        x_freq = x_freq(:); y_freq = y_freq(:);
        % converting these into a polar format, where angle is orientation and
        % magnitude is spatial frequency
        [ang_grid,mag_grid] = cart2pol(x_freq,y_freq);
        % adjust the angles so that they go from 0-pi in rads
        %ang_grid = mod(ang_grid,pi);
        ang_grid = mod(ang_grid,pi);
        ang_grid = reshape(ang_grid,final_size_pix,final_size_pix);


        % contrast before 
            image_contrast = std(ang_grid(:));


       %% Make a gaussian orientation filter in spectral domain 

%             % adjust for weird 0 deg being vertical thing
            if TrialStuff(n).orient >= 90 
               input_orient = abs(TrialStuff(n).orient-270);
            elseif TrialStuff(n).orient < 90
                input_orient = abs(TrialStuff(n).orient-90);
            end

            tar_ang = reshape(circ_vmpdf(ang_grid*2, (input_orient)*pi/180*2, TrialStuff(n).kappa*pi/180*2),final_size_pix,final_size_pix);
            
            
            
            tar_ang = tar_ang./max(tar_ang(:));
            % important - make sure that the filter has a constant value at the very center
            % of the frequency space. because this value has no angular
            % meaning.
            tar_ang(center_pix,center_pix) = 0;
      
                
            %% FFT and filter the image

            image_fft = image;
            
                  
            % apply the filters
            image_fft_filt = image_fft.*tar_ang;


            mag = abs(image_fft_filt);
            
            
            % replace phase values with random numbers between -pi +pi
            fake_phase = (rand(size(image_fft_filt))-0.5)*2*pi;

            % create the full complex array again
            image_fft_filt_shuff = complex(mag.*sin(fake_phase), mag.*cos(fake_phase));
           
            
            % back to spatial domain
            image_filtered = real(ifft2(fftshift(image_fft_filt_shuff)));
            
            
            % lost some contrast in the fourier domain, scale it back up
            current_contrast = std(image_filtered(:));
            scaling_factor = image_contrast/current_contrast;
            image_final = image_filtered*scaling_factor;
            


end
        


