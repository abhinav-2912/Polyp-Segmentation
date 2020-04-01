clear all; close all; clc;
%*****************************************************
% Processing input image I
D1 = fspecial('gaussian', 5, 15);
jpgFiles = dir('*.jpg'); % for .jpg files
numfiles = length(jpgFiles);
 
  for  p = 1:numfiles% number:number  1:numfiles%
    fprintf('%d ',p) 

I{p} = imread(jpgFiles(p).name);
 
[Ny,Nx,Nc] = size(I{p});
[x y] = meshgrid(0:1/Ny:1-1/Ny, 0:1/Nx: 1-1/Nx);
M = double(sqrt((x-0.5).^2 + (y-0.5).^2)- 0.45<= 0);

imall{p} = illumcor_zhengyu(I{p}, 0.3, 0.6, 10);

imall{p}(:,:,1) = double(imall{p}(:,:,1));
imall{p}(:,:,2) = double(imall{p}(:,:,2));
imall{p}(:,:,3) = double(imall{p}(:,:,3));

i0=Nx/2;
j0=Nx/2;

[x, y]=meshgrid(-0.5+1/Nx:1/Nx:0.5, -0.5+1/Nx:1/Nx:0.5);

% mask= double(sqrt(x.^2+y.^2)> 0.48); % experiments for 0.45, 0.48
mask= double(sqrt(x.^2+y.^2)>= 0.43);


A{p} =  (imall{p}(:,:,1) + imall{p}(:,:,2) + imall{p}(:,:,3))/3 ;

cform = makecform('srgb2lab');
lab = applycform(imall{p},cform); 
labd{p} = lab2double(lab);
     % labd1{p} = labd{p}(:,:,1); % L-channel
      labd2{p} = labd{p}(:,:,2); % a-channel
      % labd3{p} = labd{p}(:,:,3); % b-channel

 uextA{p} = mextend(double(A{p}), mask, x, y, Nx);
 
 uextB{p}  = mextend1(double(A{p}), mask, x, y, Nx);
 
 uexta{p} = mextend(labd2{p}, mask, x, y, Nx);
 
  uextab{p} = mextend1(labd2{p}, mask, x, y, Nx);
end
 
      
%  for  p = 1:1% number:number  1:numfiles%
% figure(p), 
%    subplot 321, imagesc(A{p}), colorbar, axis equal tight,  title('A channel');
%   subplot 322, imagesc(B{p}),  axis equal tight,  title('A channel with mask');
%   subplot 323, imagesc(uextA), colorbar, axis equal tight,  title('A extension');
%   subplot 324, imagesc(uextA.*M), colorbar, axis equal tight , title('A extension with mask');
%   subplot 325, imagesc(uextB), colorbar, axis equal tight,  title('A extension');
%   subplot 326, imagesc(uextB.*M), colorbar, axis equal tight , title('A extension with mask');
% %       saveas(gcf, ['normal_mucosa' ,num2str(p), '.jpg']);
%  end
 

  for  p = 1:numfiles% number:number  1:numfiles%
figure(p), subplot 321, imagesc(labd2{p}), colorbar, axis equal tight,  title('a channel');
  subplot 322, imagesc(labd2{p}.*M), axis equal tight,   title('a channel with mask');
 subplot 323, imagesc(uexta{p} ), colorbar, axis equal tight,  title('a extension with mextend');
       subplot 324, imagesc(uexta{p} .*M), colorbar, axis equal tight , title('a extension with mextend multipl with mask');
        subplot 325, imagesc(uextab{p} ), colorbar, axis equal tight,  title('a extension with mextend1');
       subplot 326, imagesc(uextab{p} .*M), colorbar, axis equal tight , title('a extension with mextend1 multipl with mask');
       saveas(gcf, ['.42' ,num2str(p), '.jpg']);
  end